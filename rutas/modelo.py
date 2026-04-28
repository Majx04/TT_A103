import os
import time
import pickle
import shutil
import numpy as np
import base64

from io import StringIO, BytesIO
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

modelo_bp = Blueprint('modelo_bp', __name__)

def _app():
    import app as _a
    return _a

def login_required():
    return "usuario" in session

# ════════════════════════════════════════════════════════════
# GET  /modelo/vista
# ════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════
# POST /modelo/entrenar
# ════════════════════════════════════════════════════════════
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

        le_local     = LabelEncoder()
        y            = le_local.fit_transform(y)
        num_classes  = len(np.unique(y))

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

        device       = a.device
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
                    pass

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

        nuevo_modelo.eval()
        with torch.no_grad():
            preds = nuevo_modelo(X_test_t.to(device)).argmax(1).cpu()
        test_acc = (preds == y_test_t).float().mean().item()

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


# ════════════════════════════════════════════════════════════
# POST /modelo/guardar
# ════════════════════════════════════════════════════════════
@modelo_bp.route("/modelo/guardar", methods=["POST"])
def modelo_guardar_version():
    if not login_required():
        return jsonify({"error": "No autenticado"}), 401

    a      = _app()
    origen = os.path.join(a.MODELO_DIR, "modelo_lstm.pt")

    if not os.path.exists(origen):
        return jsonify({"error": "No hay modelo entrenado"})

    ts      = time.strftime("%Y%m%d_%H%M%S")
    nombre  = f"modelo_{ts}.pt"
    destino = os.path.join(a.app.config['MODEL_FOLDER'], nombre)
    shutil.copy2(origen, destino)

    for arch in ["scaler.pkl", "label_encoder.pkl", "metadata.pkl"]:
        src = os.path.join(a.MODELO_DIR, arch)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(
                a.app.config['MODEL_FOLDER'],
                arch.replace(".pkl", f"_{ts}.pkl")
            ))

    return jsonify({"ok": True, "nombre": nombre})


# ════════════════════════════════════════════════════════════
# POST /modelo/analisis  ← código exacto del notebook
# ════════════════════════════════════════════════════════════
@modelo_bp.route("/modelo/analisis", methods=["POST"])
def modelo_analisis():
    if not login_required():
        return jsonify({"error": "No autenticado"}), 401

    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    archivo           = request.files.get("archivo")
    eliminar_outliers = request.form.get("eliminar_outliers") == "1"

    if not archivo:
        return jsonify({"error": "No se recibió archivo"})

    try:
        # ── Cargar CSV ────────────────────────────────────────
        df = pd.read_csv(StringIO(archivo.read().decode("utf-8")))

        if "label" not in df.columns:
            return jsonify({"error": "El CSV debe tener columna 'label'"})

        COLORES   = ['#5b8fb9', '#ef4444', '#16a34a', '#f59e0b',
                     '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']
        feat_cols = [c for c in ["WL", "RMS", "MAV", "WAMP"] if c in df.columns]
        outliers_info = None

        # ── Conteo de clases (del notebook) ──────────────────
        conteo = df["label"].value_counts()
        total  = len(df)

        print("Distribución de clases:")
        print(conteo)
        print(f"\nTotal de muestras: {total}")
        print(f"\nPorcentaje por clase:")
        print((conteo / total * 100).round(2).astype(str) + "%")

        # ── Eliminar outliers (del notebook) ─────────────────
        if eliminar_outliers and feat_cols:
            antes = len(df)
            df    = df[(np.abs(stats.zscore(df[feat_cols])) < 3).all(axis=1)]
            outliers_info = {
                "antes":      antes,
                "despues":    len(df),
                "eliminados": antes - len(df),
            }
            print(f"\nMuestras antes de limpiar: {antes}")
            print(f"Muestras después de limpiar: {len(df)}")
            print(f"Outliers eliminados: {antes - len(df)}")

            # Recalcular conteo con datos limpios
            conteo = df["label"].value_counts()
            total  = len(df)

        # ── Diccionario de clases para el frontend ────────────
        clases_dict = {}
        for i, (clase, count) in enumerate(conteo.items()):
            clases_dict[str(clase)] = {
                "count": int(count),
                "pct":   round(count / total * 100, 1),
                "color": COLORES[i % len(COLORES)],
            }

        # ── Gráfica balance (del notebook) ───────────────────
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        colores_bar = [COLORES[i % len(COLORES)] for i in range(len(conteo))]
        bars = ax1.bar(
            conteo.index.astype(str), conteo.values,
            color=colores_bar, edgecolor="black", width=0.6
        )
        ax1.set_title("Balance de clases")
        ax1.set_xlabel("Clase")
        ax1.set_ylabel("Número de muestras")
        ax1.tick_params(axis='x', rotation=0)
        for i, v in enumerate(conteo.values):
            ax1.text(i, v + 1, str(v), ha="center", fontweight="bold")
        plt.tight_layout()

        buf1 = BytesIO()
        fig1.savefig(buf1, format='png', dpi=120, bbox_inches='tight')
        buf1.seek(0)
        img_balance = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)

        # ── Pairplot (del notebook) ───────────────────────────
        paleta   = {str(c): COLORES[i % len(COLORES)]
                    for i, c in enumerate(df["label"].unique())}
        df_plot  = df[feat_cols + ["label"]].copy()
        df_plot["label"] = df_plot["label"].astype(str)

        sns.set_style("whitegrid")
        pair_grid = sns.pairplot(
            df_plot,
            hue="label",
            vars=feat_cols,
            palette=paleta,
            plot_kws={"alpha": 0.6, "s": 20},
            diag_kind="kde"
        )
        titulo = "Dispersión entre características"
        if eliminar_outliers:
            titulo += " (sin outliers)"
        pair_grid.figure.suptitle(titulo, y=1.02, fontsize=13, fontweight='bold')

        buf2 = BytesIO()
        pair_grid.figure.savefig(buf2, format='png', dpi=110, bbox_inches='tight')
        buf2.seek(0)
        img_pairplot = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close('all')

        return jsonify({
            "clases":        clases_dict,
            "total":         total,
            "img_balance":   img_balance,
            "img_pairplot":  img_pairplot,
            "outliers_info": outliers_info,
        })

    except Exception as e:
        return jsonify({"error": str(e)})