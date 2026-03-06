import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "super_secret_key_123"

# --- CONFIGURACIÓN DE BASE DE DATOS ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'pacientes.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads', 'perfiles')

db = SQLAlchemy(app)


# --- MODELOS ---

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)


class Paciente(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    edad = db.Column(db.String(20))
    sexo = db.Column(db.String(20))
    fecha_nacimiento = db.Column(db.String(20))
    nombre_madre = db.Column(db.String(100))
    enfermedades = db.Column(db.Text)
    intervenciones = db.Column(db.Text)
    responsable = db.Column(db.String(100))
    supervisores = db.Column(db.Text)
    contacto_emergencia_nom = db.Column(db.String(100))
    telefono = db.Column(db.String(20))
    observaciones = db.Column(db.Text)
    foto = db.Column(db.String(200), default='default_user.png')


# Inicialización de DB
with app.app_context():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    db.create_all()


# --- RUTAS ---

@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_input = request.form["username"]
        pass_input = request.form["password"]

        # Buscamos al administrador en la base de datos
        admin = Admin.query.filter_by(username=user_input).first()

        # Verificamos si existe y si la contraseña coincide (hash)
        if admin and check_password_hash(admin.password, pass_input):
            session["usuario"] = user_input
            return redirect(url_for("home"))
        ########Primer usuario para acceso
        else:
            # Opción de respaldo para el primer acceso si la DB está vacía
            if user_input == "admin" and pass_input == "1234":
                session["usuario"] = user_input
                return redirect(url_for("home"))

            return render_template("login.html", error="Credenciales incorrectas")

    return render_template("login.html")


@app.route("/home")
def home():
    if "usuario" not in session:
        return redirect(url_for("login"))
    return render_template("home.html")


@app.route("/logout")
def logout():
    session.pop("usuario", None)
    return redirect(url_for("login"))


@app.route("/usuario", methods=["GET", "POST"])
def usuario():
    if "usuario" not in session:
        return redirect(url_for("login"))

    paciente = Paciente.query.first()

    if request.method == "POST":
        if not paciente:
            paciente = Paciente(nombre=request.form.get("nombre"))
            db.session.add(paciente)

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

        if 'foto' in request.files:
            file = request.files['foto']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                paciente.foto = filename

        db.session.commit()
        return redirect(url_for("usuario"))

    return render_template("usuario.html", paciente=paciente)


@app.route("/nuevoAdmin", methods=["GET", "POST"])
def nuevoAdmin():
    if "usuario" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        user = request.form.get("nuevo_usuario")
        password = request.form.get("nuevo_password")

        # IMPORTANTE: Usamos 'Admin' (la clase), no 'usuario' (la función)
        usuario_existente = Admin.query.filter_by(username=user).first()

        if usuario_existente:
            flash("El nombre de usuario ya está registrado.")
            return redirect(url_for("nuevoAdmin"))

        # Encriptar contraseña de forma segura
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        nuevo_admin = Admin(username=user, password=hashed_pw)

        db.session.add(nuevo_admin)
        db.session.commit()

        flash("Administrador creado con éxito.")
        return redirect(url_for("home"))

    return render_template("nuevoAdmin.html")


if __name__ == "__main__":
    app.run(debug=True)