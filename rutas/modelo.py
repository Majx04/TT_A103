import os
import time
import pickle
import shutil
import numpy as np

from io import StringIO
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify, Response

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

modelo_bp = Blueprint('modelo_bp', __name__)

# ── Helper: importar variables globales de app sin circular ──────────────────
def _app():
    import app as _a
    return _a

def login_required():
    return "usuario" in session

# ════════════════════════════════════════════════════════════════════════════
# GET  /modelo/vista
# ════════════════════════════════════════════════════════════════════════════
@modelo_bp.route("/modelo/vista")
def modelo_vista():
    if not login_required():
        return redirect(url_for("login"))

    a = _app()
    carpeta_modelos = a.app.config['MODEL_FOLDER']
    versiones = []

    if os.path.exists(carpeta_modelos):
        for nombre in sorted(os.listdir(carpeta_modelos), reverse=True):
            if nombre.endswith(".pt"):
                ruta  = os.path.join(carpeta_modelos, nombre)
                fecha = time.strftime("%d/%m/%Y %H:%M",
                                      time.localtime(os.path.getmtime(ruta)))
                versiones.append({"nombre": nombre, "fecha": fecha})

    return render_template("modelo.html", versiones=versiones)


# ════════════════════════════════════════════════════════════════════════════
# POST /modelo/entrenar
# ════════════════════════════════════════════════════════════════════════════
@modelo_bp.route("/modelo/entrenar", methods=["POST"])
def modelo_entrenar():
    if not login_required():
        return jsonify({"error": "No autenticado"}), 401

    a = _app()

    archivo    = request.files.get("archivo")
    features   = request.form.get("features", "WL,RMS,MAV,WAMP").split(",")
    epochs     = int(request.form.get("epochs",   400))
    batch_size = int(request.form.get("batch",     16))
    lr         = float(request.form.get("lr",    0.001))
    split      = float(request.form.get("split",  0.2))
    reentrenar = request.form.get("reentrenar", "0") == "1"

    if not archivo:
        return jsonify({"error": "No se recibió archivo"})

    try:
        import pandas as pd

        df = pd.read_csv(StringIO(archivo.read().decode("utf-8")))

        for feat in features:
            if feat not in df.columns:
                return jsonify({"error": f"Columna '{feat}' no encontrada en el CSV"})
        if "label" not in df.columns:
            return jsonify({"error": "El CSV debe tener una columna 'label'"})

        X = df[features].values
        y = df["label"].values

        le_local = LabelEncoder()
        y        = le_local.fit_transform(y)
        num_classes = len(np.unique(y))

        scaler_local = StandardScaler()
        X = scaler_local.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=42, shuffle=True
        )

        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t  = torch.tensor(y_test,  dtype=torch.long)

        val_size   = int(0.2 * len(X_train_t))
        train_size = len(X_train_t) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            TensorDataset(X_train_t, y_train_t),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        device = a.device

        # Cargar pesos existentes si es reentrenamiento
        nuevo_modelo = a.LSTMClassifier(
            input_size=len(features),
            num_classes=num_classes
        ).to(device)

        if reentrenar:
            pt_path = os.path.join(a.MODELO_DIR, "modelo_lstm.pt")
            if os.path.exists(pt_path):
                try:
                    ckpt = torch.load(pt_path, map_location=device)
                    nuevo_modelo.load_state_dict(ckpt["model_state_dict"])
                except Exception:
                    pass  # arquitectura diferente → empieza desde cero

        optimizer = optim.Adam(nuevo_modelo.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_epoch   = 1

        for epoch in range(1, epochs + 1):
            nuevo_modelo.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(nuevo_modelo(X_b), y_b)
                loss.backward()
                optimizer.step()

            nuevo_modelo.eval()
            val_correct, val_total, val_loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    out = nuevo_modelo(X_b)
                    val_loss_sum += criterion(out, y_b).item() * len(y_b)
                    val_correct  += (out.argmax(1) == y_b).sum().item()
                    val_total    += len(y_b)

            val_acc = val_correct / val_total
            scheduler.step(val_loss_sum / val_total)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch

        # Evaluación final en test
        nuevo_modelo.eval()
        with torch.no_grad():
            preds = nuevo_modelo(X_test_t.to(device)).argmax(1).cpu()
        test_acc = (preds == y_test_t).float().mean().item()

        # Guardar modelo activo
        os.makedirs(a.MODELO_DIR, exist_ok=True)
        torch.save({
            "model_state_dict": nuevo_modelo.state_dict(),
            "input_size":       len(features),
            "num_classes":      num_classes,
        }, os.path.join(a.MODELO_DIR, "modelo_lstm.pt"))

        with open(os.path.join(a.MODELO_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler_local, f)
        with open(os.path.join(a.MODELO_DIR, "label_encoder.pkl"), "wb") as f:
            pickle.dump(le_local, f)

        meta = {
            "feature_cols":      features,
            "num_classes":       num_classes,
            "classes":           list(le_local.classes_),
            "test_accuracy":     float(test_acc),
            "best_epoch":        int(best_epoch),
            "best_val_accuracy": float(best_val_acc),
            "epochs":            epochs,
            "batch_size":        batch_size,
        }
        with open(os.path.join(a.MODELO_DIR, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)

        # Actualizar variables globales en app.py
        a.modelo_lstm    = nuevo_modelo
        a.scaler         = scaler_local
        a.le             = le_local
        a.metadata       = meta
        a.MODELO_CARGADO = True

        return jsonify({
            "test_accuracy":     float(test_acc),
            "best_val_accuracy": float(best_val_acc),
            "best_epoch":        int(best_epoch),
            "classes":           list(le_local.classes_),
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ════════════════════════════════════════════════════════════════════════════
# POST /modelo/guardar
# ════════════════════════════════════════════════════════════════════════════
@modelo_bp.route("/modelo/guardar", methods=["POST"])
def modelo_guardar_version():
    if not login_required():
        return jsonify({"error": "No autenticado"}), 401

    a = _app()
    origen = os.path.join(a.MODELO_DIR, "modelo_lstm.pt")

    if not os.path.exists(origen):
        return jsonify({"error": "No hay modelo entrenado"})

    ts      = time.strftime("%Y%m%d_%H%M%S")
    nombre  = f"modelo_{ts}.pt"
    destino = os.path.join(a.app.config['MODEL_FOLDER'], nombre)
    shutil.copy2(origen, destino)

    # Copiar artefactos con el mismo timestamp
    for arch in ["scaler.pkl", "label_encoder.pkl", "metadata.pkl"]:
        src = os.path.join(a.MODELO_DIR, arch)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(
                a.app.config['MODEL_FOLDER'],
                arch.replace(".pkl", f"_{ts}.pkl")
            ))

    return jsonify({"ok": True, "nombre": nombre})


# ════════════════════════════════════════════════════════════════════════════
# POST /modelo/analisis
# ════════════════════════════════════════════════════════════════════════════
@modelo_bp.route("/modelo/analisis", methods=["POST"])
def modelo_analisis():
    if not login_required():
        return jsonify({"error": "No autenticado"}), 401

    archivo = request.files.get("archivo")
    if not archivo:
        return jsonify({"error": "No se recibió archivo"})

    try:
        import pandas as pd

        df = pd.read_csv(StringIO(archivo.read().decode("utf-8")))

        if "label" not in df.columns:
            return jsonify({"error": "El CSV debe tener columna 'label'"})

        feat_cols = [c for c in ["WL", "RMS", "MAV", "WAMP"] if c in df.columns]

        # Conteo por clase
        conteo = df["label"].value_counts().to_dict()

        # Puntos para dispersión (máx 500)
        muestra = df.sample(min(500, len(df)), random_state=42)
        puntos  = muestra[feat_cols + ["label"]].to_dict(orient="records")

        # Estadísticas por clase
        stats_clases = {}
        for clase, grupo in df.groupby("label"):
            stats_clases[str(clase)] = {
                feat: {
                    "mean": float(grupo[feat].mean()),
                    "std":  float(grupo[feat].std()),
                    "min":  float(grupo[feat].min()),
                    "max":  float(grupo[feat].max()),
                }
                for feat in feat_cols
            }

        return jsonify({
            "conteo_clases": conteo,
            "puntos":        puntos,
            "stats_clases":  stats_clases,
            "features":      feat_cols,
        })

    except Exception as e:
        return jsonify({"error": str(e)})