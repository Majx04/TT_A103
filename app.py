import os
import atexit
import serial
import threading
import collections
import numpy as np
import csv
import time
import pickle
import shutil
import json
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from io import StringIO

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========================
# APP
# ========================
app = Flask(__name__)
app.secret_key = "super_secret_key_123"

# ========================
# CONFIG
# ========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'pacientes.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER']  = os.path.join(BASE_DIR, 'static', 'uploads', 'perfiles')
app.config['DATASET_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads', 'datasets')
app.config['MODEL_FOLDER']   = os.path.join(BASE_DIR, 'static', 'models')

for folder in [app.config['UPLOAD_FOLDER'],
               app.config['DATASET_FOLDER'],
               app.config['MODEL_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

db = SQLAlchemy(app)

# ========================
# MODELO LSTM (PyTorch)
# ========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.lstm1   = nn.LSTM(input_size, 256, batch_first=True)
        self.bn1     = nn.BatchNorm1d(256)
        self.lstm2   = nn.LSTM(256, 128, batch_first=True)
        self.bn2     = nn.BatchNorm1d(128)
        self.lstm3   = nn.LSTM(128, 64, batch_first=True)
        self.bn3     = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.1)
        self.fc1     = nn.Linear(64, 64)
        self.bn4     = nn.BatchNorm1d(64)
        self.fc2     = nn.Linear(64, 32)
        self.fc3     = nn.Linear(32, num_classes)
        self.relu    = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.bn1(out[:, -1, :]).unsqueeze(1)
        out, _ = self.lstm2(out)
        out = self.bn2(out[:, -1, :]).unsqueeze(1)
        out, _ = self.lstm3(out)
        out = self.bn3(out[:, -1, :])
        out = self.dropout(out)
        out = self.relu(self.bn4(self.fc1(out)))
        out = self.relu(self.fc2(out))
        return self.fc3(out)

# ========================
# CARGAR MODELO Y ARTEFACTOS
# ========================
MODELO_DIR = os.path.join(BASE_DIR, "modelo_guardado")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clasificacion_actual = {"etiqueta": "—", "confianza": 0.0}
clasificacion_lock   = threading.Lock()

try:
    checkpoint = torch.load(
        os.path.join(MODELO_DIR, "modelo_lstm.pt"),
        map_location=device
    )
    modelo_lstm = LSTMClassifier(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"]
    ).to(device)
    modelo_lstm.load_state_dict(checkpoint["model_state_dict"])
    modelo_lstm.eval()

    scaler   = pickle.load(open(os.path.join(MODELO_DIR, "scaler.pkl"),        "rb"))
    le       = pickle.load(open(os.path.join(MODELO_DIR, "label_encoder.pkl"), "rb"))
    metadata = pickle.load(open(os.path.join(MODELO_DIR, "metadata.pkl"),      "rb"))

    MODELO_CARGADO = True
    print("✅ Modelo LSTM cargado correctamente")
    print(f"   Clases: {metadata['classes']}")
except Exception as e:
    modelo_lstm    = None
    scaler         = None
    le             = None
    metadata       = {}
    MODELO_CARGADO = False
    print(f"⚠️  No se pudo cargar el modelo: {e}")

# ========================
# EXTRACCIÓN DE CARACTERÍSTICAS
# ========================
def extraer_caracteristicas(segmento):
    seg  = np.array(segmento, dtype=np.float64)
    WL   = np.sum(np.abs(np.diff(seg)))
    RMS  = np.sqrt(np.mean(seg ** 2))
    MAV  = np.mean(np.abs(seg))
    WAMP = np.sum(np.abs(np.diff(seg)) > 0.01)
    return np.array([[WL, RMS, MAV, WAMP]], dtype=np.float64)

# ========================
# CLASIFICAR SEGMENTO
# ========================
def clasificar_segmento(segmento):
    if not MODELO_CARGADO:
        return
    try:
        feats   = extraer_caracteristicas(segmento)
        feats_n = scaler.transform(feats)
        feats_r = feats_n.reshape(1, 1, feats_n.shape[1])
        tensor  = torch.tensor(feats_r, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = modelo_lstm(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx      = int(np.argmax(probs))
        etiqueta = le.inverse_transform([idx])[0]
        conf     = float(probs[idx])

        with clasificacion_lock:
            clasificacion_actual["etiqueta"]  = str(etiqueta)
            clasificacion_actual["confianza"] = conf
    except Exception as e:
        print(f"Error clasificando: {e}")

# ========================
# EMG CONFIG
# ========================
PUERTO   = 'COM3'
BAUDRATE = 921600

FS       = 1000
VENTANA  = 5
MUESTRAS = FS * VENTANA

VREF    = 3.3
ADC_MAX = 4095.0

data_emg = collections.deque([0.0] * MUESTRAS, maxlen=MUESTRAS)

pausado           = False
contador_muestras = 0
carpeta_emg       = os.path.join(BASE_DIR, "static", "emg")
os.makedirs(carpeta_emg, exist_ok=True)

# ========================
# SERIAL — CIERRE LIMPIO AL SALIR
# ========================
ser      = None
ser_lock = threading.Lock()

def cerrar_serial():
    """Libera el puerto al cerrar Flask con Ctrl+C."""
    global ser
    with ser_lock:
        if ser is not None and ser.is_open:
            try:
                ser.close()
                print(f"✅ Puerto {PUERTO} cerrado correctamente")
            except Exception as e:
                print(f"⚠️  Error al cerrar puerto: {e}")

atexit.register(cerrar_serial)

# ========================
# CONEXIÓN SERIAL ROBUSTA
# ========================
def conectar_serial():
    """Intenta abrir el puerto serial. Reintenta cada 3s si falla."""
    global ser
    while True:
        with ser_lock:
            if ser is None or not ser.is_open:
                try:
                    if ser is not None:
                        try:
                            ser.close()
                        except:
                            pass
                    s   = serial.Serial(PUERTO, BAUDRATE, timeout=1)
                    ser = s
                    print(f"✅ Puerto {PUERTO} abierto correctamente")
                except serial.SerialException as e:
                    ser = None
                    print(f"⚠️  No se pudo abrir {PUERTO}: {e} — reintentando en 3s...")
        time.sleep(3)

# ========================
# HILO SERIAL
# ========================
def leer_serial():
    global ser
    while True:
        if pausado:
            time.sleep(0.01)
            continue

        with ser_lock:
            s = ser

        if s is None or not s.is_open:
            time.sleep(0.1)
            continue

        try:
            linea = s.readline().decode("utf-8", errors="ignore").strip()
            if not linea:
                continue
            try:
                valor   = float(linea)
                voltaje = valor * VREF / ADC_MAX if valor > 10 else valor
                data_emg.append(voltaje)
            except ValueError:
                pass
        except serial.SerialException as e:
            print(f"❌ Error serial: {e} — reconectando...")
            with ser_lock:
                try:
                    ser.close()
                except:
                    pass
                ser = None
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error inesperado en serial: {e}")
            time.sleep(0.1)

# ========================
# HILO DE CLASIFICACIÓN PERIÓDICA
# ========================
def hilo_clasificacion():
    while True:
        time.sleep(0.5)
        if not pausado:
            segmento = list(data_emg)
            threading.Thread(
                target=clasificar_segmento,
                args=(segmento,),
                daemon=True
            ).start()

# Arranca los tres hilos
threading.Thread(target=conectar_serial,    daemon=True).start()
threading.Thread(target=leer_serial,        daemon=True).start()
threading.Thread(target=hilo_clasificacion, daemon=True).start()

# ========================
# DB MODELS
# ========================
class Admin(db.Model):
    id                = db.Column(db.Integer, primary_key=True)
    username          = db.Column(db.String(50), unique=True, nullable=False)
    password          = db.Column(db.String(255), nullable=False)
    can_manage_admins = db.Column(db.Boolean, default=False)

class Paciente(db.Model):
    id                      = db.Column(db.Integer, primary_key=True)
    nombre                  = db.Column(db.String(100))
    edad                    = db.Column(db.String(20))
    sexo                    = db.Column(db.String(20))
    fecha_nacimiento        = db.Column(db.String(50))
    nombre_madre            = db.Column(db.String(100))
    enfermedades            = db.Column(db.Text)
    intervenciones          = db.Column(db.Text)
    responsable             = db.Column(db.String(100))
    supervisores            = db.Column(db.String(100))
    contacto_emergencia_nom = db.Column(db.String(100))
    telefono                = db.Column(db.String(20))
    observaciones           = db.Column(db.Text)
    foto                    = db.Column(db.String(200), default='default_user.png')

with app.app_context():
    db.create_all()

# ========================
# UTIL
# ========================
def login_required():
    return "usuario" in session

# ========================
# RUTAS
# ========================
@app.route("/")
def index():
    return redirect(url_for("login"))

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_input = request.form["username"]
        pass_input = request.form["password"]
        admin = Admin.query.filter_by(username=user_input).first()
        if admin and check_password_hash(admin.password, pass_input):
            session["usuario"]    = user_input
            session["can_manage"] = admin.can_manage_admins
            return redirect(url_for("home"))
        elif user_input == "admin" and pass_input == "1234":
            session["usuario"]    = user_input
            session["can_manage"] = True
            return redirect(url_for("home"))
        return render_template("login.html", error="Credenciales incorrectas")
    return render_template("login.html")

# ---------- HOME ----------
@app.route("/home")
def home():
    if not login_required():
        return redirect(url_for("login"))
    admin_actual = Admin.query.filter_by(username=session["usuario"]).first()
    return render_template("home.html", admin=admin_actual)

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.pop("usuario", None)
    return redirect(url_for("login"))

# ---------- USUARIO ----------
@app.route("/usuario", methods=["GET", "POST"])
def usuario():
    if not login_required():
        return redirect(url_for("login"))
    paciente = Paciente.query.first()
    if not paciente:
        paciente = Paciente(nombre="Nuevo Paciente")
        db.session.add(paciente)
        db.session.commit()
    if request.method == "POST":
        paciente.nombre                  = request.form.get("nombre")
        paciente.edad                    = request.form.get("edad")
        paciente.sexo                    = request.form.get("sexo")
        paciente.fecha_nacimiento        = request.form.get("fecha_nacimiento")
        paciente.nombre_madre            = request.form.get("nombre_madre")
        paciente.enfermedades            = request.form.get("enfermedades")
        paciente.intervenciones          = request.form.get("intervenciones")
        paciente.responsable             = request.form.get("responsable")
        paciente.supervisores            = request.form.get("supervisores")
        paciente.contacto_emergencia_nom = request.form.get("contacto_emergencia_nom")
        paciente.telefono                = request.form.get("telefono")
        paciente.observaciones           = request.form.get("observaciones")
        file = request.files.get("foto")
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            paciente.foto = filename
        db.session.commit()
        flash("Información actualizada.")
        return redirect(url_for("usuario"))
    return render_template("usuario.html", paciente=paciente)

# ---------- NUEVO ADMIN ----------
@app.route("/nuevoAdmin", methods=["GET", "POST"])
def nuevoAdmin():
    if not login_required():
        return redirect(url_for("login"))
    if not session.get("can_manage"):
        flash("Permisos insuficientes")
        return redirect(url_for("home"))
    if request.method == "POST":
        user     = request.form.get("nuevo_usuario")
        password = request.form.get("nuevo_password")
        p_admin  = request.form.get("perm_admin") == "on"
        if Admin.query.filter_by(username=user).first():
            flash("Usuario ya existe")
        else:
            db.session.add(Admin(
                username=user,
                password=generate_password_hash(password),
                can_manage_admins=p_admin
            ))
            db.session.commit()
            flash("Administrador creado")
            return redirect(url_for("gestionar_admins"))
    return render_template("nuevoAdmin.html")

# ---------- GESTIÓN ADMINS ----------
@app.route("/gestionar_admins", methods=["GET", "POST"])
def gestionar_admins():
    if not login_required():
        return redirect(url_for("login"))
    if not session.get("can_manage"):
        return redirect(url_for("home"))
    if request.method == "POST":
        admin_id = request.args.get('id')
        if admin_id:
            admin = Admin.query.get(admin_id)
            if admin:
                admin.can_manage_admins = request.form.get("p_admin") == "on"
                db.session.commit()
                flash("Permisos actualizados")
    admins = Admin.query.all()
    return render_template("gestionar_admins.html", admins=admins)

# ---------- ELIMINAR ADMIN ----------
@app.route("/eliminar_admin/<int:id>")
def eliminar_admin(id):
    if not login_required():
        return redirect(url_for("login"))
    if not session.get("can_manage"):
        return redirect(url_for("home"))
    admin = Admin.query.get_or_404(id)
    if admin.username == session.get("usuario"):
        flash("No puedes eliminarte a ti mismo")
        return redirect(url_for("gestionar_admins"))
    db.session.delete(admin)
    db.session.commit()
    flash("Administrador eliminado")
    return redirect(url_for("gestionar_admins"))

# ---------- EMG ----------
@app.route("/emg")
def emg():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("adquisicion.html")

@app.route("/emg/data")
def emg_data():
    return jsonify(list(data_emg))

@app.route("/emg/toggle", methods=["POST"])
def emg_toggle():
    global pausado
    pausado = not pausado
    return jsonify({"pausado": pausado})

@app.route("/emg/guardar", methods=["POST"])
def emg_guardar():
    global contador_muestras
    print(">>> form data:", dict(request.form))
    if not pausado:
        return jsonify({"error": "Debes pausar antes de guardar"})

    nombre = request.form.get("nombre", "").strip()
    label  = request.form.get("label",  "").strip()

    if not nombre:
        return jsonify({"error": "Nombre vacío"})
    if not label:
        return jsonify({"error": "Selecciona una etiqueta"})

    contador_muestras += 1
    nombre_final = f"{nombre}_{contador_muestras}"
    ventana      = list(data_emg)
    ruta_csv     = os.path.join(carpeta_emg, f"{nombre_final}.csv")
    ruta_png     = os.path.join(carpeta_emg, f"{nombre_final}.png")

    # ── CSV ──────────────────────────────────────────────
    with open(ruta_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["indice", "voltaje", "label"])
        for i, v in enumerate(ventana):
            writer.writerow([i, v, label])
    print(f"✅ CSV guardado: {ruta_csv}")

    # ── PNG ──────────────────────────────────────────────
    try:
        datos_validos = [v for v in ventana if v != 0.0]
        print(f"   Muestras totales: {len(ventana)} | No-cero: {len(datos_validos)}")
        print(f"   Max: {max(ventana):.4f} | Min: {min(ventana):.4f}")

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(ventana, linewidth=0.8, color="#00e5ff")
        ax.set_ylim(0, 3.5)
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Señal EMG — {nombre_final} [{label}]", color="white", fontsize=11)
        ax.set_xlabel("Muestra", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Voltaje (V)", color="#94a3b8", fontsize=9)

        fig.savefig(ruta_png, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        if os.path.exists(ruta_png):
            print(f"✅ PNG guardado: {ruta_png} ({os.path.getsize(ruta_png)} bytes)")
        else:
            print(f"❌ PNG NO se creó en: {ruta_png}")

    except Exception as e:
        print(f"❌ Error generando PNG: {e}")
        import traceback
        traceback.print_exc()

    return jsonify({"ok": nombre_final})

@app.route("/emg/clasificacion")
def emg_clasificacion():
    with clasificacion_lock:
        return jsonify({
            "etiqueta":  clasificacion_actual["etiqueta"],
            "confianza": clasificacion_actual["confianza"]
        })

@app.route("/emg/clases")
def emg_clases():
    clases = metadata.get("classes", []) if MODELO_CARGADO else []
    return jsonify({"clases": clases})

# ---------- DEBUG (eliminar en producción) ----------
@app.route("/emg/debug")
def emg_debug():
    muestra = list(data_emg)
    return jsonify({
        "serial_disponible": ser is not None,
        "pausado":           pausado,
        "primeros_10":       muestra[:10],
        "max":               max(muestra),
        "min":               min(muestra),
        "todos_cero":        all(v == 0.0 for v in muestra)
    })

# ---------- IA ----------
@app.route("/entrenar")
def entrenar():
    data_iris = load_iris()
    X = data_iris.data
    y = data_iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return jsonify({
        "accuracy":         float(accuracy_score(y_test, pred)),
        "precision":        float(precision_score(y_test, pred, average="macro")),
        "recall":           float(recall_score(y_test, pred, average="macro")),
        "f1":               float(f1_score(y_test, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    })

@app.route("/modelo")
def modelo_view():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("vistaModelo.html")

# ---------- CSV COMBINAR ----------

@app.route("/csv", methods=["GET"])
def csv_route():
    if not login_required():
        return redirect(url_for("login"))

    carpeta  = os.path.join(BASE_DIR, "static", "emg")
    archivos = []

    if os.path.exists(carpeta):
        for nombre in sorted(os.listdir(carpeta)):
            if nombre.endswith(".csv"):
                archivos.append({"nombre": nombre})

    return render_template("csv.html", archivos=archivos)

# ---------- HISTORIAL ----------
@app.route("/historial")
def historial():
    if not login_required():
        return redirect(url_for("login"))

    carpeta  = os.path.join(BASE_DIR, "static", "emg")
    archivos = []

    if os.path.exists(carpeta):
        for nombre in sorted(os.listdir(carpeta), reverse=True):
            if nombre.endswith(".csv") or nombre.endswith(".png"):
                ruta  = os.path.join(carpeta, nombre)
                tipo  = "csv" if nombre.endswith(".csv") else "png"
                fecha = time.strftime(
                    "%d/%m/%Y %H:%M",
                    time.localtime(os.path.getmtime(ruta))
                )
                archivos.append({
                    "nombre": nombre,
                    "tipo":   tipo,
                    "fecha":  fecha,
                })

    return render_template("historial.html", archivos=archivos)

# ---------- ELIMINAR MUESTRA (independiente por archivo) ----------
@app.route("/historial/eliminar/<nombre>", methods=["POST"])
def eliminar_muestra(nombre):
    if not login_required():
        return jsonify({"error": "No autorizado"}), 401

    nombre  = os.path.basename(nombre)
    carpeta = os.path.join(BASE_DIR, "static", "emg")
    ruta    = os.path.join(carpeta, nombre)

    if os.path.exists(ruta):
        os.remove(ruta)
        return jsonify({"ok": True})
    else:
        return jsonify({"error": "Archivo no encontrado"}), 404

# ---------- EXTRAER CARACTERÍSTICAS ----------
# ── Reemplaza la ruta /extraer en app.py ─────────────────
# Añade al inicio de app.py si no están:
#   from scipy import signal as scipy_signal
#   from scipy.fft import fft, fftfreq

from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

# ── Config ────────────────────────────────────────────────
WINDOW_SIZE = 200
STEP        = 50
THRESHOLD   = 0.01
FS          = 1000
BAND_LOW    = (0,   60)
BAND_MID    = (60,  150)
BAND_HIGH   = (150, 500)

# ── Funciones temporales ──────────────────────────────────
def feat_MAV(sig):
    return float(np.mean(np.abs(sig)))

def feat_WL(sig):
    return float(np.sum(np.abs(np.diff(sig))))

def feat_WAMP(sig, thr=THRESHOLD):
    return int(np.sum(np.abs(np.diff(sig)) > thr))

def feat_VAR(sig):
    return float(np.var(sig))

def feat_SNR(sig):
    mean_sig = np.mean(sig)
    noise    = np.var(sig - mean_sig)
    return float(10 * np.log10(mean_sig**2 / noise)) if noise > 0 else 0.0

def feat_ZC(sig, thr=THRESHOLD):
    signs     = np.sign(sig)
    crossings = np.where(np.diff(signs) != 0)[0]
    return int(np.sum(np.abs(np.diff(sig)[crossings]) > thr))

def feat_SSC(sig, thr=THRESHOLD):
    d1 = np.diff(sig[:-1]); d2 = np.diff(sig[1:])
    return int(np.sum(((d1 * d2) < 0) & ((np.abs(d1) > thr) | (np.abs(d2) > thr))))

def feat_LOG(sig):
    a = np.abs(sig); a = np.where(a < 1e-10, 1e-10, a)
    return float(np.exp(np.mean(np.log(a))))

def feat_SSI(sig):
    return float(np.sum(sig ** 2))

# ── Funciones frecuenciales ───────────────────────────────
def _spectrum(sig):
    freqs = fftfreq(len(sig), d=1/FS)
    power = np.abs(fft(sig)) ** 2
    h = len(freqs) // 2
    return freqs[:h], power[:h]

def feat_MedFreq(sig):
    freqs, power = _spectrum(sig)
    cum = np.cumsum(power); total = cum[-1]
    if total == 0: return 0.0
    return float(freqs[min(np.searchsorted(cum, total/2), len(freqs)-1)])

def feat_MeanFreq(sig):
    freqs, power = _spectrum(sig)
    total = np.sum(power)
    return float(np.sum(freqs * power) / total) if total > 0 else 0.0

def feat_SpectralEntropy(sig):
    _, power = _spectrum(sig)
    total = np.sum(power)
    if total == 0: return 0.0
    p = power / total; p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def feat_SpectralKurtosis(sig):
    freqs, power = _spectrum(sig)
    total = np.sum(power)
    if total == 0: return 0.0
    p = power / total
    mf = np.sum(freqs * p)
    sf = np.sqrt(np.sum(((freqs - mf)**2) * p))
    return float(np.sum(((freqs - mf)**4) * p) / sf**4) if sf > 0 else 0.0

def feat_TopFreqs(sig, n=5):
    freqs, power = _spectrum(sig)
    idx = np.argsort(power)[::-1][:n]
    result = list(freqs[idx])
    while len(result) < n: result.append(0.0)
    return result

def feat_E_Total(sig):
    _, power = _spectrum(sig); return float(np.sum(power))

def feat_band_energy(sig, f_low, f_high):
    freqs, power = _spectrum(sig)
    return float(np.sum(power[(freqs >= f_low) & (freqs < f_high)]))

def feat_TFR(sig):
    _, _, Zxx = scipy_signal.stft(sig, fs=FS, nperseg=min(64, len(sig)//2))
    mag = np.abs(Zxx)
    pn  = mag / (np.sum(mag) + 1e-10)
    return float(np.mean(mag**2)), float(np.max(mag**2)), float(-np.sum(pn * np.log(pn + 1e-10)))

# ── Ruta Flask ────────────────────────────────────────────
@app.route("/extraer", methods=["GET", "POST"])
def extraer():
    if not login_required():
        return redirect(url_for("login"))

    if request.method == "POST":
        archivos          = request.files.getlist("archivos")
        features_pedidas  = set(request.form.getlist("features"))  # features seleccionadas

        if not archivos or all(f.filename == "" for f in archivos):
            flash("No se seleccionaron archivos")
            return redirect(url_for("extraer"))

        if not features_pedidas:
            flash("Selecciona al menos una característica")
            return redirect(url_for("extraer"))

        # Orden canónico de columnas (solo las seleccionadas)
        TODAS = [
            "MAV", "WL", "WAMP", "VAR", "SNR",
            "ZC", "SSC", "LOG", "SSI",
            "MedFreq", "MeanFreq", "SpectralEntropy", "SpectralKurtosis",
            "TopFreq1", "TopFreq2", "TopFreq3", "TopFreq4", "TopFreq5",
            "E_Total", "E_Low", "E_Mid", "E_High",
            "R_Low", "R_Mid", "R_High", "R_HighLow",
            "TFR_MeanEnergy", "TFR_MaxEnergy", "TFR_Entropy",
            "label"
        ]

        # Las top5 se agrupan bajo "TopFreqs" en el form pero se expanden a 5 cols
        def col_activa(col):
            if col in ("TopFreq1","TopFreq2","TopFreq3","TopFreq4","TopFreq5"):
                return "TopFreqs" in features_pedidas
            if col == "label":
                return True
            return col in features_pedidas

        columnas = [c for c in TODAS if col_activa(c)]

        salida = StringIO()
        writer = csv.writer(salida)
        writer.writerow(columnas)

        for f in archivos:
            if not f.filename.endswith(".csv"):
                continue
            try:
                contenido = f.read().decode("utf-8").splitlines()
                reader    = csv.DictReader(contenido)
                filas     = list(reader)
                if not filas:
                    continue

                col_volt  = next((c for c in filas[0] if c.strip().lower() == "voltaje"), None)
                col_label = next((c for c in filas[0] if c.strip().lower() == "label"),   None)
                if col_volt is None:
                    continue

                voltajes, etiquetas = [], []
                for fila in filas:
                    try:
                        voltajes.append(float(fila[col_volt]))
                        etiquetas.append(fila[col_label].strip() if col_label else "")
                    except (ValueError, KeyError):
                        pass

                if len(voltajes) < WINDOW_SIZE:
                    continue

                sig_full = np.array(voltajes, dtype=np.float64)

                for i in range(0, len(sig_full) - WINDOW_SIZE, STEP):
                    w        = sig_full[i:i + WINDOW_SIZE]
                    w_labels = etiquetas[i:i + WINDOW_SIZE]

                    # Calcular todo (solo se escribe lo seleccionado)
                    e_total = feat_E_Total(w)
                    e_low   = feat_band_energy(w, *BAND_LOW)
                    e_mid   = feat_band_energy(w, *BAND_MID)
                    e_high  = feat_band_energy(w, *BAND_HIGH)
                    tfr_mean, tfr_max, tfr_e = feat_TFR(w)
                    top5 = feat_TopFreqs(w)

                    from collections import Counter
                    label = Counter(w_labels).most_common(1)[0][0] if w_labels else ""

                    todas_vals = {
                        "MAV":              feat_MAV(w),
                        "WL":               feat_WL(w),
                        "WAMP":             feat_WAMP(w),
                        "VAR":              feat_VAR(w),
                        "SNR":              feat_SNR(w),
                        "ZC":               feat_ZC(w),
                        "SSC":              feat_SSC(w),
                        "LOG":              feat_LOG(w),
                        "SSI":              feat_SSI(w),
                        "MedFreq":          feat_MedFreq(w),
                        "MeanFreq":         feat_MeanFreq(w),
                        "SpectralEntropy":  feat_SpectralEntropy(w),
                        "SpectralKurtosis": feat_SpectralKurtosis(w),
                        "TopFreq1": top5[0], "TopFreq2": top5[1],
                        "TopFreq3": top5[2], "TopFreq4": top5[3], "TopFreq5": top5[4],
                        "E_Total":       e_total,
                        "E_Low":         e_low,
                        "E_Mid":         e_mid,
                        "E_High":        e_high,
                        "R_Low":         e_low  / e_total if e_total > 0 else 0,
                        "R_Mid":         e_mid  / e_total if e_total > 0 else 0,
                        "R_High":        e_high / e_total if e_total > 0 else 0,
                        "R_HighLow":     e_high / e_low   if e_low   > 0 else 0,
                        "TFR_MeanEnergy": tfr_mean,
                        "TFR_MaxEnergy":  tfr_max,
                        "TFR_Entropy":    tfr_e,
                        "label":          label,
                    }

                    writer.writerow([todas_vals[c] for c in columnas])

            except Exception as e:
                print(f"Error procesando {f.filename}: {e}")
                import traceback; traceback.print_exc()
                continue

        lineas = salida.getvalue().strip().splitlines()
        if len(lineas) <= 1:
            flash("No se pudieron extraer características de los archivos")
            return redirect(url_for("extraer"))

        return Response(
            salida.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=caracteristicas.csv"}
        )

    return render_template("extraer.html")

# ---------- MODELO BLUEPRINT ----------
from rutas.modelo import modelo_bp
app.register_blueprint(modelo_bp)



# ========================
# MAIN
# ========================
if __name__ == "__main__":
    # use_reloader=False evita el doble proceso que bloquea el puerto serial
    app.run(debug=True, use_reloader=False)