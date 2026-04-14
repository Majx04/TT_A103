import os
import serial
import threading
import collections
import numpy as np
import csv
import time
import pickle

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
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

try:
    ser = serial.Serial(PUERTO, BAUDRATE, timeout=1)
    print(f"✅ Puerto {PUERTO} abierto")
except Exception as e:
    ser = None
    print(f"⚠️  Serial no disponible: {e}")

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
# HILO SERIAL
# ========================
def leer_serial():
    if ser is None:
        return
    while True:
        if pausado:
            time.sleep(0.01)
            continue
        try:
            linea = ser.readline().decode().strip()
            if linea.isdigit():
                adc     = int(linea)
                voltaje = adc * VREF / ADC_MAX
                data_emg.append(voltaje)
        except:
            pass

threading.Thread(target=leer_serial, daemon=True).start()

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
    if not pausado:
        return jsonify({"error": "Debes pausar antes de guardar"})
    nombre = request.form.get("nombre")
    if not nombre:
        return jsonify({"error": "Nombre vacío"})
    contador_muestras += 1
    nombre_final = f"{nombre}_{contador_muestras}"
    ventana      = list(data_emg)
    ruta_csv = os.path.join(carpeta_emg, f"{nombre_final}.csv")
    ruta_png = os.path.join(carpeta_emg, f"{nombre_final}.png")
    with open(ruta_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["indice", "voltaje"])
        for i, v in enumerate(ventana):
            writer.writerow([i, v])
    plt.figure()
    plt.plot(ventana)
    plt.ylim(0, 4)
    plt.title("EMG")
    plt.savefig(ruta_png)
    plt.close()
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

@app.route("/historial")
def historial():
    if not login_required():
        return redirect(url_for("login"))

    carpeta = os.path.join(BASE_DIR, "static", "emg")
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

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    app.run(debug=True)