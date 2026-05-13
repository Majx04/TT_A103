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
app.config['MODEL_FOLDER']   = os.path.join(BASE_DIR, 'static', 'modelos')

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
# Imports adicionales necesarios al inicio de app.py:
#   from scipy import signal as scipy_signal
#   from scipy.fft import fft, fftfreq
#   import pandas as pd   (si no está ya)

from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
WINDOW_SIZE = 200
STEP        = 50
THRESHOLD   = 0.01
FS          = 1000
BAND_LOW    = (0,   60)
BAND_MID    = (60,  150)
BAND_HIGH   = (150, 500)

# ══════════════════════════════════════════════════════════
#  FUNCIONES TEMPORALES  (igual que el script .py original)
# ══════════════════════════════════════════════════════════
def _MAV(sig):
    return float(np.mean(np.abs(sig)))

def _WL(sig):
    return float(np.sum(np.abs(np.diff(sig))))

def _WAMP(sig):
    return int(np.sum(np.abs(np.diff(sig)) > THRESHOLD))

def _VAR(sig):
    return float(np.var(sig))

def _SNR(sig):
    mean_sig     = np.mean(sig)
    signal_power = mean_sig ** 2
    noise_power  = np.var(sig - mean_sig)
    if noise_power == 0:
        return 0.0
    return float(10 * np.log10(signal_power / noise_power))

def _ZC(sig):
    signs     = np.sign(sig)
    crossings = np.where(np.diff(signs) != 0)[0]
    return int(np.sum(np.abs(np.diff(sig)[crossings]) > THRESHOLD))

def _SSC(sig):
    diff1 = np.diff(sig[:-1])
    diff2 = np.diff(sig[1:])
    sign_change  = (diff1 * diff2) < 0
    amplitude_ok = (np.abs(diff1) > THRESHOLD) | (np.abs(diff2) > THRESHOLD)
    return int(np.sum(sign_change & amplitude_ok))

def _LOG(sig):
    abs_sig = np.abs(sig)
    abs_sig = np.where(abs_sig < 1e-10, 1e-10, abs_sig)
    return float(np.exp(np.mean(np.log(abs_sig))))

def _SSI(sig):
    return float(np.sum(sig ** 2))

# ══════════════════════════════════════════════════════════
#  FUNCIONES FRECUENCIALES  (igual que el script .py original)
# ══════════════════════════════════════════════════════════
def _get_spectrum(sig):
    freqs = fftfreq(len(sig), d=1 / FS)
    power = np.abs(fft(sig)) ** 2
    half  = len(freqs) // 2
    return freqs[:half], power[:half]

def _median_frequency(sig):
    freqs, power = _get_spectrum(sig)
    cumulative   = np.cumsum(power)
    total        = cumulative[-1]
    if total == 0:
        return 0.0
    idx = np.searchsorted(cumulative, total / 2)
    return float(freqs[min(idx, len(freqs) - 1)])

def _TFR_features(sig):
    _, _, Zxx      = scipy_signal.stft(sig, fs=FS, nperseg=min(64, len(sig) // 2))
    magnitude      = np.abs(Zxx)
    tfr_mean       = float(np.mean(magnitude ** 2))
    tfr_max        = float(np.max(magnitude ** 2))
    power_norm     = magnitude / (np.sum(magnitude) + 1e-10)
    tfr_entropy    = float(-np.sum(power_norm * np.log(power_norm + 1e-10)))
    return tfr_mean, tfr_max, tfr_entropy

def _band_energy(sig, f_low, f_high):
    freqs, power = _get_spectrum(sig)
    mask = (freqs >= f_low) & (freqs < f_high)
    return float(np.sum(power[mask]))

def _total_power(sig):
    _, power = _get_spectrum(sig)
    return float(np.sum(power))

def _top_n_frequencies(sig, n=5):
    freqs, power = _get_spectrum(sig)
    top_idx = np.argsort(power)[::-1][:n] if len(power) >= n else np.argsort(power)[::-1]
    result  = list(freqs[top_idx])
    while len(result) < n:
        result.append(0.0)
    return result

def _spectral_entropy(sig):
    _, power = _get_spectrum(sig)
    total    = np.sum(power)
    if total == 0:
        return 0.0
    p_norm = power / total
    p_norm = p_norm[p_norm > 0]
    return float(-np.sum(p_norm * np.log2(p_norm)))

def _spectral_kurtosis(sig):
    freqs, power = _get_spectrum(sig)
    total = np.sum(power)
    if total == 0:
        return 0.0
    p_norm = power / total
    mean_f = np.sum(freqs * p_norm)
    std_f  = np.sqrt(np.sum(((freqs - mean_f) ** 2) * p_norm))
    if std_f == 0:
        return 0.0
    return float(np.sum(((freqs - mean_f) ** 4) * p_norm) / (std_f ** 4))

def _mean_frequency(sig):
    freqs, power = _get_spectrum(sig)
    total = np.sum(power)
    return float(np.sum(freqs * power) / total) if total > 0 else 0.0

# ══════════════════════════════════════════════════════════
#  RUTA FLASK
# ══════════════════════════════════════════════════════════
# ── Reemplaza la ruta /extraer en app.py ─────────────────
# Imports adicionales necesarios al inicio de app.py:
#   from scipy import signal as scipy_signal
#   from scipy.fft import fft, fftfreq

from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import io as _io
import pandas as _pd

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
WINDOW_SIZE = 200
STEP        = 50
THRESHOLD   = 0.01
FS          = 1000
BAND_LOW    = (0,   60)
BAND_MID    = (60,  150)
BAND_HIGH   = (150, 500)

# ══════════════════════════════════════════════════════════
#  FUNCIONES TEMPORALES
# ══════════════════════════════════════════════════════════
def _MAV(sig):
    return float(np.mean(np.abs(sig)))

def _WL(sig):
    return float(np.sum(np.abs(np.diff(sig))))

def _WAMP(sig):
    return int(np.sum(np.abs(np.diff(sig)) > THRESHOLD))

def _VAR(sig):
    return float(np.var(sig))

def _SNR(sig):
    mean_sig     = np.mean(sig)
    signal_power = mean_sig ** 2
    noise_power  = np.var(sig - mean_sig)
    if noise_power == 0:
        return 0.0
    return float(10 * np.log10(signal_power / noise_power))

def _ZC(sig):
    signs     = np.sign(sig)
    crossings = np.where(np.diff(signs) != 0)[0]
    return int(np.sum(np.abs(np.diff(sig)[crossings]) > THRESHOLD))

def _SSC(sig):
    diff1 = np.diff(sig[:-1])
    diff2 = np.diff(sig[1:])
    sign_change  = (diff1 * diff2) < 0
    amplitude_ok = (np.abs(diff1) > THRESHOLD) | (np.abs(diff2) > THRESHOLD)
    return int(np.sum(sign_change & amplitude_ok))

def _LOG(sig):
    abs_sig = np.abs(sig)
    abs_sig = np.where(abs_sig < 1e-10, 1e-10, abs_sig)
    return float(np.exp(np.mean(np.log(abs_sig))))

def _SSI(sig):
    return float(np.sum(sig ** 2))

# ══════════════════════════════════════════════════════════
#  FUNCIONES FRECUENCIALES
# ══════════════════════════════════════════════════════════
def _get_spectrum(sig):
    freqs = fftfreq(len(sig), d=1 / FS)
    power = np.abs(fft(sig)) ** 2
    half  = len(freqs) // 2
    return freqs[:half], power[:half]

def _median_frequency(sig):
    freqs, power = _get_spectrum(sig)
    cumulative   = np.cumsum(power)
    total        = cumulative[-1]
    if total == 0:
        return 0.0
    idx = np.searchsorted(cumulative, total / 2)
    return float(freqs[min(idx, len(freqs) - 1)])

def _TFR_features(sig):
    _, _, Zxx   = scipy_signal.stft(sig, fs=FS, nperseg=min(64, len(sig) // 2))
    magnitude   = np.abs(Zxx)
    tfr_mean    = float(np.mean(magnitude ** 2))
    tfr_max     = float(np.max(magnitude ** 2))
    power_norm  = magnitude / (np.sum(magnitude) + 1e-10)
    tfr_entropy = float(-np.sum(power_norm * np.log(power_norm + 1e-10)))
    return tfr_mean, tfr_max, tfr_entropy

def _band_energy(sig, f_low, f_high):
    freqs, power = _get_spectrum(sig)
    mask = (freqs >= f_low) & (freqs < f_high)
    return float(np.sum(power[mask]))

def _total_power(sig):
    _, power = _get_spectrum(sig)
    return float(np.sum(power))

def _top_n_frequencies(sig, n=5):
    freqs, power = _get_spectrum(sig)
    top_idx = np.argsort(power)[::-1][:n] if len(power) >= n else np.argsort(power)[::-1]
    result  = list(freqs[top_idx])
    while len(result) < n:
        result.append(0.0)
    return result

def _spectral_entropy(sig):
    _, power = _get_spectrum(sig)
    total    = np.sum(power)
    if total == 0:
        return 0.0
    p_norm = power / total
    p_norm = p_norm[p_norm > 0]
    return float(-np.sum(p_norm * np.log2(p_norm)))

def _spectral_kurtosis(sig):
    freqs, power = _get_spectrum(sig)
    total = np.sum(power)
    if total == 0:
        return 0.0
    p_norm = power / total
    mean_f = np.sum(freqs * p_norm)
    std_f  = np.sqrt(np.sum(((freqs - mean_f) ** 2) * p_norm))
    if std_f == 0:
        return 0.0
    return float(np.sum(((freqs - mean_f) ** 4) * p_norm) / (std_f ** 4))

def _mean_frequency(sig):
    freqs, power = _get_spectrum(sig)
    total = np.sum(power)
    return float(np.sum(freqs * power) / total) if total > 0 else 0.0

def _dominant_label(w_labels):
    """
    Retorna la etiqueta dominante en la ventana.
    Usa el mismo criterio que el script original: mode()[0].
    Maneja arrays vacíos, todos-NaN y strings vacíos.
    """
    serie = _pd.Series(w_labels)
    # Filtrar vacíos y NaN
    serie = serie[serie.notna() & (serie.astype(str).str.strip() != "")]
    if serie.empty:
        return ""
    moda = serie.mode()
    return str(moda.iloc[0]) if not moda.empty else str(serie.iloc[0])

# ══════════════════════════════════════════════════════════
#  RUTA FLASK
# ══════════════════════════════════════════════════════════
# ── Reemplaza la ruta /extraer en app.py ─────────────────
# Imports adicionales al inicio de app.py si no están:
#   from scipy import signal as scipy_signal
#   from scipy.fft import fft, fftfreq
#   import pandas as pd
#   import io

from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import io as _io
import pandas as _pd

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
WINDOW_SIZE = 200
STEP        = 50
THRESHOLD   = 0.01
FS          = 1000
BAND_LOW    = (0,   60)
BAND_MID    = (60,  150)
BAND_HIGH   = (150, 500)

# ══════════════════════════════════════════════════════════
#  FUNCIONES TEMPORALES
# ══════════════════════════════════════════════════════════
def _MAV(sig):
    return float(np.mean(np.abs(sig)))

def _WL(sig):
    return float(np.sum(np.abs(np.diff(sig))))

def _WAMP(sig):
    return int(np.sum(np.abs(np.diff(sig)) > THRESHOLD))

def _VAR(sig):
    return float(np.var(sig))

def _SNR(sig):
    mean_sig    = np.mean(sig)
    noise_power = np.var(sig - mean_sig)
    if noise_power == 0:
        return 0.0
    return float(10 * np.log10(mean_sig ** 2 / noise_power))

def _ZC(sig):
    signs     = np.sign(sig)
    crossings = np.where(np.diff(signs) != 0)[0]
    return int(np.sum(np.abs(np.diff(sig)[crossings]) > THRESHOLD))

def _SSC(sig):
    d1 = np.diff(sig[:-1]); d2 = np.diff(sig[1:])
    return int(np.sum(((d1 * d2) < 0) & ((np.abs(d1) > THRESHOLD) | (np.abs(d2) > THRESHOLD))))

def _LOG(sig):
    a = np.abs(sig); a = np.where(a < 1e-10, 1e-10, a)
    return float(np.exp(np.mean(np.log(a))))

def _SSI(sig):
    return float(np.sum(sig ** 2))

# ══════════════════════════════════════════════════════════
#  FUNCIONES FRECUENCIALES
# ══════════════════════════════════════════════════════════
def _get_spectrum(sig):
    freqs = fftfreq(len(sig), d=1/FS)
    power = np.abs(fft(sig)) ** 2
    h     = len(freqs) // 2
    return freqs[:h], power[:h]

def _median_frequency(sig):
    freqs, power = _get_spectrum(sig)
    cum   = np.cumsum(power); total = cum[-1]
    if total == 0: return 0.0
    return float(freqs[min(np.searchsorted(cum, total/2), len(freqs)-1)])

def _TFR_features(sig):
    _, _, Zxx  = scipy_signal.stft(sig, fs=FS, nperseg=min(64, len(sig)//2))
    mag        = np.abs(Zxx)
    pn         = mag / (np.sum(mag) + 1e-10)
    return float(np.mean(mag**2)), float(np.max(mag**2)), float(-np.sum(pn * np.log(pn + 1e-10)))

def _band_energy(sig, f_low, f_high):
    freqs, power = _get_spectrum(sig)
    return float(np.sum(power[(freqs >= f_low) & (freqs < f_high)]))

def _total_power(sig):
    _, power = _get_spectrum(sig); return float(np.sum(power))

def _top_n_frequencies(sig, n=5):
    freqs, power = _get_spectrum(sig)
    idx    = np.argsort(power)[::-1][:n] if len(power) >= n else np.argsort(power)[::-1]
    result = list(freqs[idx])
    while len(result) < n: result.append(0.0)
    return result

def _spectral_entropy(sig):
    _, power = _get_spectrum(sig); total = np.sum(power)
    if total == 0: return 0.0
    p = power / total; p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def _spectral_kurtosis(sig):
    freqs, power = _get_spectrum(sig); total = np.sum(power)
    if total == 0: return 0.0
    p  = power / total
    mf = np.sum(freqs * p)
    sf = np.sqrt(np.sum(((freqs - mf)**2) * p))
    return float(np.sum(((freqs - mf)**4) * p) / sf**4) if sf > 0 else 0.0

def _mean_frequency(sig):
    freqs, power = _get_spectrum(sig); total = np.sum(power)
    return float(np.sum(freqs * power) / total) if total > 0 else 0.0

# ══════════════════════════════════════════════════════════
#  RUTA FLASK
# ══════════════════════════════════════════════════════════
@app.route("/extraer", methods=["GET", "POST"])
def extraer():
    if not login_required():
        return redirect(url_for("login"))

    if request.method == "POST":
        archivos         = request.files.getlist("archivos")
        features_pedidas = set(request.form.getlist("features"))

        if not archivos or all(f.filename == "" for f in archivos):
            flash("No se seleccionaron archivos")
            return redirect(url_for("extraer"))

        if not features_pedidas:
            flash("Selecciona al menos una característica")
            return redirect(url_for("extraer"))

        # Orden canónico
        TODAS = [
            "MAV", "WL", "WAMP", "VAR", "SNR",
            "ZC", "SSC", "LOG", "SSI",
            "MedFreq",
            "TFR_MeanEnergy", "TFR_MaxEnergy", "TFR_Entropy",
            "E_Total", "E_Low", "E_Mid", "E_High",
            "R_Low", "R_Mid", "R_High", "R_HighLow",
            "TopFreq1", "TopFreq2", "TopFreq3", "TopFreq4", "TopFreq5",
            "SpectralEntropy", "SpectralKurtosis", "MeanFreq",
            "label"
        ]

        def col_activa(col):
            if col in ("TopFreq1","TopFreq2","TopFreq3","TopFreq4","TopFreq5"):
                return "TopFreqs" in features_pedidas
            if col == "label":
                return True          # label SIEMPRE se incluye
            return col in features_pedidas

        columnas = [c for c in TODAS if col_activa(c)]

        salida = StringIO()
        writer = csv.writer(salida)
        writer.writerow(columnas)

        for f in archivos:
            if not f.filename.endswith(".csv"):
                continue
            try:
                contenido = f.read().decode("utf-8")
                df_arch   = _pd.read_csv(_io.StringIO(contenido))

                # Normalizar nombres de columnas (quitar espacios)
                df_arch.columns = [c.strip() for c in df_arch.columns]

                print(f">>> {f.filename} — columnas: {list(df_arch.columns)}")

                # Buscar columna voltaje
                col_volt = next(
                    (c for c in df_arch.columns if c.lower() == "voltaje"), None
                )
                if col_volt is None:
                    print(f"  ✗ Sin columna 'voltaje' — omitiendo")
                    continue

                # Buscar columna label — acepta label, extension, flexion
                LABEL_NAMES = {"label", "extension", "flexion", "etiqueta"}
                col_label = next(
                    (c for c in df_arch.columns if c.strip().lower() in LABEL_NAMES), None
                )
                print(f"  col_volt={col_volt!r}  col_label={col_label!r}")

                sig_full = df_arch[col_volt].astype(float).values

                if col_label is not None:
                    raw = df_arch[col_label].astype(str).str.strip()
                    if col_label.strip().lower() in {"extension", "flexion"}:
                        labels_full = np.full(len(sig_full), col_label.strip().lower(), dtype=object)
                        print(f"  Columna-etiqueta: {col_label!r} asignada como label fijo")
                    else:
                        labels_full = raw.values
                    print(f"  Primeras etiquetas: {labels_full[:5]}")
                else:
                    labels_full = np.full(len(sig_full), "", dtype=object)
                    print(f"  ✗ Sin columna label")

                if len(sig_full) < WINDOW_SIZE:
                    print(f"  ✗ Señal demasiado corta ({len(sig_full)} muestras)")
                    continue

                ventanas_procesadas = 0
                for i in range(0, len(sig_full) - WINDOW_SIZE, STEP):
                    w        = sig_full[i:i + WINDOW_SIZE]
                    w_labels = labels_full[i:i + WINDOW_SIZE]

                    # Temporales
                    mav  = _MAV(w);   wl   = _WL(w)
                    wamp = _WAMP(w);  var  = _VAR(w);  snr  = _SNR(w)
                    zc   = _ZC(w);    ssc  = _SSC(w)
                    log  = _LOG(w);   ssi  = _SSI(w)

                    # Frecuenciales
                    mf                       = _median_frequency(w)
                    tfr_mean, tfr_max, tfr_e = _TFR_features(w)
                    e_total                  = _total_power(w)
                    e_low                    = _band_energy(w, *BAND_LOW)
                    e_mid                    = _band_energy(w, *BAND_MID)
                    e_high                   = _band_energy(w, *BAND_HIGH)
                    r_low                    = e_low  / e_total if e_total > 0 else 0
                    r_mid                    = e_mid  / e_total if e_total > 0 else 0
                    r_high                   = e_high / e_total if e_total > 0 else 0
                    r_hi_lo                  = e_high / e_low   if e_low   > 0 else 0
                    top5                     = _top_n_frequencies(w, n=5)
                    sp_entropy               = _spectral_entropy(w)
                    sp_kurtosis              = _spectral_kurtosis(w)
                    mean_freq                = _mean_frequency(w)

                    # ── Etiqueta dominante: mode()[0] igual que el script original ──
                    serie   = _pd.Series(w_labels)
                    moda    = serie.mode()
                    label   = moda.iloc[0] if len(moda) > 0 else ""

                    todas_vals = {
                        "MAV": mav, "WL": wl, "WAMP": wamp, "VAR": var, "SNR": snr,
                        "ZC": zc, "SSC": ssc, "LOG": log, "SSI": ssi,
                        "MedFreq": mf,
                        "TFR_MeanEnergy": tfr_mean, "TFR_MaxEnergy": tfr_max, "TFR_Entropy": tfr_e,
                        "E_Total": e_total, "E_Low": e_low, "E_Mid": e_mid, "E_High": e_high,
                        "R_Low": r_low, "R_Mid": r_mid, "R_High": r_high, "R_HighLow": r_hi_lo,
                        "TopFreq1": top5[0], "TopFreq2": top5[1], "TopFreq3": top5[2],
                        "TopFreq4": top5[3], "TopFreq5": top5[4],
                        "SpectralEntropy": sp_entropy, "SpectralKurtosis": sp_kurtosis,
                        "MeanFreq": mean_freq,
                        "label": label,
                    }

                    writer.writerow([todas_vals[c] for c in columnas])
                    ventanas_procesadas += 1

                print(f"  ✔ {ventanas_procesadas} ventanas procesadas")

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

#----DISEÑO MECANICO
from flask import render_template

@app.route('/diseno-mecanico')
def diseno_mecanico():
    return render_template('mecanico.html')


# ========================
# MAIN
# ========================
if __name__ == "__main__":
    # use_reloader=False evita el doble proceso que bloquea el puerto serial
    app.run(debug=True, use_reloader=False)