"""
routes/arbol_route.py
======================
Blueprint Flask para entrenar el Random Forest desde la interfaz web.
Guarda resultados en JSON para no reentrenar en cada refresh.

Registrar en app.py:
    from routes.arbol_route import arbol_bp
    app.register_blueprint(arbol_bp)
"""

import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from flask import (Blueprint, jsonify, render_template,
                   request, send_file, session)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

arbol_bp = Blueprint('arbol', __name__)

# ── Archivos persistentes ────────────────────────────────────────────────────
RESULTS_PATH = os.path.join('static', 'arbol_results.json')
SAVE_DIR     = os.path.join('static', 'modelo_rf_guardado')

FEATURE_COLS = [
    "WAMP", "VAR",
    "ZC", "SSC", "LOG", "SSI",
    "MedFreq",
    "TFR_MeanEnergy", "TFR_MaxEnergy", "TFR_Entropy",
    "E_Total", "E_Low", "E_Mid", "E_High",
    "R_Low", "R_Mid", "R_High",
    "R_HighLow",
    "TopFreq1", "TopFreq2", "TopFreq3", "TopFreq4", "TopFreq5",
    "SpectralEntropy", "SpectralKurtosis", "MeanFreq",
]

# ── Páginas ──────────────────────────────────────────────────────────────────

@arbol_bp.route('/arbol')
def arbol():
    return render_template('arbol.html')


@arbol_bp.route('/arbol/resultados')
def vista_arbol():
    resultados = _cargar_resultados()
    return render_template('vista_arbol.html', resultados=resultados)


@arbol_bp.route('/arbol/descargar')
def arbol_download():
    model_path = os.path.join(SAVE_DIR, 'modelo_rf.pkl')
    if not os.path.exists(model_path):
        return 'modelo_rf.pkl no encontrado. Entrena el modelo primero.', 404
    return send_file(model_path, as_attachment=True, download_name='modelo_rf.pkl')


# ── Entrenamiento ────────────────────────────────────────────────────────────

@arbol_bp.route('/arbol/train', methods=['POST'])
def arbol_train():
    file = request.files.get('csv_file')
    if not file:
        return jsonify({'error': 'No se recibió ningún archivo CSV'}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'No se pudo leer el CSV: {e}'}), 400

    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        return jsonify({'error': f'Columnas faltantes: {missing}'}), 400
    if 'label' not in df.columns:
        return jsonify({'error': 'Columna "label" no encontrada'}), 400

    # 1. Preparar datos
    X = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df['label'].values

    mask = np.isfinite(X).all(axis=1)
    X, y_raw = X[mask], y_raw[mask]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    n_classes = len(classes)

    # 2. Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    # 4. GridSearchCV
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth':    [None, 20, 40],
        'max_features': ['sqrt', 'log2'],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1,
                               class_weight='balanced'),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    rf = grid.best_estimator_
    best_p = grid.best_params_

    # 5. Validación cruzada
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5,
                                scoring='accuracy', n_jobs=-1)

    # 6. Evaluar en test
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 7. Classification report
    report_dict = classification_report(y_test, y_pred,
                                        target_names=[str(c) for c in classes],
                                        output_dict=True)
    report_rows = []
    for cls_name in [str(c) for c in classes]:
        v = report_dict.get(cls_name, {})
        report_rows.append({
            'label':     cls_name,
            'precision': v.get('precision', 0),
            'recall':    v.get('recall', 0),
            'f1':        v.get('f1-score', 0),
            'support':   int(v.get('support', 0)),
        })

    # 8. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred).tolist()

    # 9. Importancia de features
    imps = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    imps_sorted = imps.sort_values(ascending=False)
    feat_imp = [(k, float(v)) for k, v in imps_sorted.items()]

    # 10. Guardar artefactos
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, 'modelo_rf.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    with open(os.path.join(SAVE_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(SAVE_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    metadata = {
        'feature_cols':        FEATURE_COLS,
        'num_classes':         n_classes,
        'classes':             classes,
        'test_accuracy':       float(acc),
        'oob_score':           float(rf.oob_score_) if rf.oob_score else 0.0,
        'cv_mean':             float(cv_scores.mean()),
        'cv_std':              float(cv_scores.std()),
        'n_estimators':        rf.n_estimators,
        'max_depth':           str(rf.max_depth),
        'max_features':        rf.max_features,
        'feature_importances': imps_sorted.to_dict(),
    }
    with open(os.path.join(SAVE_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # 11. Guardar resultados JSON
    resultados = {
        'timestamp':           datetime.now().strftime('%d/%m/%Y %H:%M'),
        'total_samples':       int(X.shape[0]),
        'accuracy':            float(acc),
        'cv_mean':             float(cv_scores.mean()),
        'cv_std':              float(cv_scores.std()),
        'oob_score':           float(rf.oob_score_) if rf.oob_score else 0.0,
        'n_classes':           n_classes,
        'classes':             classes,
        'best_params':         best_p,
        'feature_importances': feat_imp,
        'report_rows':         report_rows,
        'confusion_matrix':    cm,
    }
    _guardar_resultados(resultados)

    return jsonify({'ok': True, 'accuracy': float(acc)})


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cargar_resultados():
    if not os.path.exists(RESULTS_PATH):
        return None
    try:
        with open(RESULTS_PATH, 'r') as f:
            data = json.load(f)
        data['feature_importances'] = [tuple(x) for x in data['feature_importances']]
        return data
    except Exception:
        return None


def _guardar_resultados(data):
    os.makedirs('static', exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)