import os

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


app = Flask(__name__)
app.secret_key = "super_secret_key_123"

modelo = None


# ---------------- CONFIGURACIÓN ----------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'pacientes.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads', 'perfiles')
app.config['DATASET_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads', 'datasets')
app.config['MODEL_FOLDER'] = os.path.join(BASE_DIR, 'static', 'models')

for folder in [
    app.config['UPLOAD_FOLDER'],
    app.config['DATASET_FOLDER'],
    app.config['MODEL_FOLDER']
]:
    os.makedirs(folder, exist_ok=True)

db = SQLAlchemy(app)


# ---------------- MODELOS DB ----------------

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    can_manage_admins = db.Column(db.Boolean, default=False)


class Paciente(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100))
    edad = db.Column(db.String(20))
    sexo = db.Column(db.String(20))
    fecha_nacimiento = db.Column(db.String(50))
    nombre_madre = db.Column(db.String(100))
    enfermedades = db.Column(db.Text)
    intervenciones = db.Column(db.Text)
    responsable = db.Column(db.String(100))
    supervisores = db.Column(db.String(100))
    contacto_emergencia_nom = db.Column(db.String(100))
    telefono = db.Column(db.String(20))
    observaciones = db.Column(db.Text)
    foto = db.Column(db.String(200), default='default_user.png')


with app.app_context():
    db.create_all()


# ---------------- UTILIDADES ----------------

def login_required():
    if "usuario" not in session:
        return False
    return True


# ---------------- RUTAS ----------------

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

        # Caso 1: Administrador de la Base de Datos
        if admin and check_password_hash(admin.password, pass_input):
            session["usuario"] = user_input
            # Guardamos si este admin específico tiene permisos de gestión
            session["can_manage"] = admin.can_manage_admins
            return redirect(url_for("home"))

        # Caso 2: Super Administrador "Hardcodeado"
        elif user_input == "admin" and pass_input == "1234":
            session["usuario"] = user_input
            session["can_manage"] = True  # El admin maestro siempre puede gestionar
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


# ---------- PACIENTE ----------

from werkzeug.utils import secure_filename

@app.route("/usuario", methods=["GET", "POST"])
def usuario():
    if not login_required():
        return redirect(url_for("login"))

    # Intentamos obtener el paciente
    paciente = Paciente.query.first()

    # SI NO EXISTE, LO CREAMOS (Solución al AttributeError)
    if not paciente:
        paciente = Paciente(nombre="Nuevo Paciente")
        db.session.add(paciente)
        db.session.commit()

    if request.method == "POST":
        # Capturamos los datos del formulario
        paciente.nombre = request.form.get("nombre")
        paciente.edad = request.form.get("edad")
        paciente.sexo = request.form.get("sexo")
        paciente.fecha_nacimiento = request.form.get("fecha_nacimiento")
        paciente.nombre_madre = request.form.get("nombre_madre")
        paciente.enfermedades = request.form.get("enfermedades")
        paciente.intervenciones = request.form.get("intervenciones")
        paciente.responsable = request.form.get("responsable")
        paciente.supervisores = request.form.get("supervisores")
        paciente.contacto_emergencia_nom = request.form.get("contacto_emergencia_nom")
        paciente.telefono = request.form.get("telefono")
        paciente.observaciones = request.form.get("observaciones")

        # Manejo de la Foto
        file = request.files.get("foto")
        if file and file.filename != '':
            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            paciente.foto = filename

        db.session.commit()
        flash("Información actualizada.")
        return redirect(url_for("usuario"))

    return render_template("usuario.html", paciente=paciente)


# ---------- ADMIN NUEVO ----------

@app.route("/nuevoAdmin", methods=["GET", "POST"])
def nuevoAdmin():
    # 1. Verificar si está logueado
    if not login_required():
        return redirect(url_for("login"))

    # 2. Verificar permisos desde la sesión (permite entrar al usuario "admin" 1234)
    if not session.get("can_manage"):
        flash("Permisos insuficientes para crear administradores.")
        return redirect(url_for("home"))

    if request.method == "POST":
        user = request.form.get("nuevo_usuario")
        password = request.form.get("nuevo_password")
        # El checkbox devuelve 'on' si está marcado, lo convertimos a Booleano
        p_admin = True if request.form.get("perm_admin") == "on" else False

        # Validar si ya existe
        if Admin.query.filter_by(username=user).first():
            flash("Error: El nombre de usuario ya está registrado.")
        else:
            hashed_pw = generate_password_hash(password)
            nuevo = Admin(
                username=user,
                password=hashed_pw,
                can_manage_admins=p_admin
            )
            db.session.add(nuevo)
            db.session.commit()
            flash(f"Administrador '{user}' creado con éxito.")
            return redirect(url_for("gestionar_admins"))

    return render_template("nuevoAdmin.html")


# ---------- GESTIÓN ADMINS ----------

@app.route("/gestionar_admins", methods=["GET", "POST"]) # Agregamos POST
def gestionar_admins():
    if not login_required():
        return redirect(url_for("login"))

    if not session.get("can_manage"):
        return redirect(url_for("home"))

    # --- LÓGICA PARA ACTUALIZAR AL PRESIONAR GUARDAR ---
    if request.method == "POST":
        admin_id = request.args.get('id') # Obtenemos el ID del admin desde la URL
        if admin_id:
            admin_a_editar = Admin.query.get(admin_id)
            if admin_a_editar:
                # Si el checkbox está marcado llega como "on", si no, llega como None
                nuevo_permiso = True if request.form.get("p_admin") == "on" else False
                admin_a_editar.can_manage_admins = nuevo_permiso
                db.session.commit()
                flash(f"Permisos de {admin_a_editar.username} actualizados.")
            return redirect(url_for("gestionar_admins"))

    # --- LÓGICA PARA MOSTRAR LA TABLA (GET) ---
    admins = Admin.query.all()
    return render_template("gestionar_admins.html", admins=admins)

@app.route("/eliminar_admin/<int:id>")
def eliminar_admin(id):
    if not login_required():
        return redirect(url_for("login"))

    # 1. Verificar que quien intenta borrar sea un administrador autorizado
    if not session.get("can_manage"):
        flash("No tienes permisos para realizar esta acción.")
        return redirect(url_for("home"))

    # 2. Buscar el administrador a eliminar
    admin_a_eliminar = Admin.query.get_or_404(id)

    # 3. Evitar que un administrador de la DB se borre a sí mismo
    if admin_a_eliminar.username == session.get("usuario"):
        flash("No puedes eliminar tu propia cuenta de la base de datos.")
        return redirect(url_for("gestionar_admins"))

    # 4. Ejecutar el borrado
    try:
        db.session.delete(admin_a_eliminar)
        db.session.commit()
        flash(f"Administrador '{admin_a_eliminar.username}' eliminado con éxito.")
    except Exception as e:
        db.session.rollback() # Si algo falla, deshacemos el cambio
        flash("Error al eliminar el administrador.")
        print(f"Error: {e}")

    return redirect(url_for("gestionar_admins"))


# ---------- MODELO IA ----------

@app.route("/modelo")
def modelo():

    if not login_required():
        return redirect(url_for("login"))

    return render_template("modelo.html")


# ---------- ENTRENAMIENTO RANDOM FOREST ----------

@app.route("/entrenar", methods=["GET","POST"])
def entrenar():

    global modelo

    data = load_iris()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="macro")
    recall = recall_score(y_test, pred, average="macro")
    f1 = f1_score(y_test, pred, average="macro")

    cm = confusion_matrix(y_test, pred)

    return jsonify({
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    })


# ---------------- MAIN ----------------

if __name__ == "__main__":
    app.run(debug=True)