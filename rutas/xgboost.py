"""
routes/xgboost_route.py
========================
"""

import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from flask import (Blueprint, jsonify, render_template,
                   request, send_file, session)
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

xgboost_bp = Blueprint('xgboost', __name__)

# ── Rutas de archivos persistentes ──────────────────────────────────────────
RESULTS_PATH = os.path.join('static', 'xgboost_results.json')
MODEL_H_PATH = os.path.join('static', 'model.h')

ALL_FEATURES = [
    "MAV", "WL", "WAMP", "VAR", "SNR",
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

@xgboost_bp.route('/xgboost')
def xgboost():
    return render_template('xgboost.html')


@xgboost_bp.route('/xgboost/resultados')
def vista_xgboost():
    resultados = _cargar_resultados()
    return render_template('vista_xgboost.html',
                           resultados=resultados,
                           enumerate=enumerate)


@xgboost_bp.route('/xgboost/descargar')
def xgboost_download():
    if not os.path.exists(MODEL_H_PATH):
        return 'model.h no encontrado. Entrena el modelo primero.', 404
    return send_file(MODEL_H_PATH, as_attachment=True, download_name='model.h')


# ── Entrenamiento ────────────────────────────────────────────────────────────

@xgboost_bp.route('/xgboost/train', methods=['POST'])
def xgboost_train():
    file = request.files.get('csv_file')
    if not file:
        return jsonify({'error': 'No se recibió ningún archivo CSV'}), 400

    # 1. Cargar datos
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'No se pudo leer el CSV: {e}'}), 400

    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        return jsonify({'error': f'Columnas faltantes: {missing}'}), 400
    if 'label' not in df.columns:
        return jsonify({'error': 'Columna "label" no encontrada'}), 400

    X = df[ALL_FEATURES].values.astype(np.float32)
    y_raw = df['label'].values

    mask = np.isfinite(X).all(axis=1)
    X, y_raw = X[mask], y_raw[mask]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    n_classes = len(classes)

    # 2. Split + normalización
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. GridSearchCV
    param_grid = {
        'n_estimators':     [100, 200],
        'max_depth':        [3, 4, 5],
        'learning_rate':    [0.05, 0.1],
        'subsample':        [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
    }
    grid = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42),
        param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    xgb = grid.best_estimator_

    # 4. Selección de features
    selector = SelectFromModel(xgb, threshold='median')
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    sel_idx = np.where(selector.get_support())[0]
    sel_names = [ALL_FEATURES[i] for i in sel_idx]

    # 5. Reentrenar con features seleccionadas
    best_p = grid.best_params_
    xgb_final = XGBClassifier(
        n_estimators=best_p['n_estimators'],
        max_depth=best_p['max_depth'],
        learning_rate=best_p['learning_rate'],
        subsample=best_p['subsample'],
        colsample_bytree=best_p['colsample_bytree'],
        eval_metric='logloss',
        random_state=42,
    )
    xgb_final.fit(X_train_sel, y_train)

    y_pred = xgb_final.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)

    # 6. Classification report
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

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Importancia de features
    imps = xgb_final.feature_importances_
    feat_imp = sorted(zip(sel_names, imps.tolist()), key=lambda x: -x[1])

    # 7. Exportar model.h
    model_h = _generar_model_h(xgb_final, scaler, sel_idx, sel_names,
                                n_classes, classes, best_p, acc)
    with open(MODEL_H_PATH, 'w') as f:
        f.write(model_h)

    # 8. Guardar resultados JSON
    resultados = {
        'timestamp':            datetime.now().strftime('%d/%m/%Y %H:%M'),
        'total_samples':        int(X.shape[0]),
        'accuracy':             float(acc),
        'n_features_sel':       len(sel_names),
        'n_trees':              len(xgb_final.get_booster().get_dump()),
        'n_classes':            n_classes,
        'classes':              [str(c) for c in classes],
        'best_params':          best_p,
        'selected_features':    sel_names,
        'feature_importances':  feat_imp,
        'report_rows':          report_rows,
        'confusion_matrix':     cm,
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
        # Convertir listas a list-of-tuples para Jinja2
        data['feature_importances'] = [tuple(x) for x in data['feature_importances']]
        return data
    except Exception:
        return None


def _guardar_resultados(data):
    os.makedirs('static', exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Generación de model.h ────────────────────────────────────────────────────

NODE_RE = re.compile(r'(\d+):\[f(\d+)<([\deE.+\-]+)\]\s+yes=(\d+),no=(\d+)')
LEAF_RE = re.compile(r'(\d+):leaf=([\deE.+\-]+)')


def _parse_tree(tree_str, feat_names, tree_idx):
    nodes = {}
    for line in tree_str.strip().split('\n'):
        line = line.strip()
        m = NODE_RE.search(line)
        if m:
            nid, feat, thr, yes, no = m.groups()
            nodes[int(nid)] = {'feat': int(feat), 'thr': float(thr),
                               'yes': int(yes), 'no': int(no),
                               'is_leaf': False, 'val': 0.0}
        else:
            m = LEAF_RE.search(line)
            if m:
                nid, val = m.groups()
                nodes[int(nid)] = {'feat': 0, 'thr': 0.0,
                                   'yes': 0, 'no': 0,
                                   'is_leaf': True, 'val': float(val)}

    lines = [f'static inline float xgb_tree_{tree_idx}(const float *x) {{']

    def recurse(nid, depth):
        indent = '    ' * depth
        nd = nodes[nid]
        if nd['is_leaf']:
            lines.append(f'{indent}return {nd["val"]:.8e}f;')
        else:
            fname = feat_names[nd['feat']]
            lines.append(f'{indent}if (x[{nd["feat"]}] < {nd["thr"]:.8e}f) {{  /* {fname} */')
            recurse(nd['yes'], depth + 1)
            lines.append(f'{indent}}} else {{')
            recurse(nd['no'],  depth + 1)
            lines.append(f'{indent}}}')

    recurse(0, 1)
    lines.append('}')
    return '\n'.join(lines)


def _generar_model_h(xgb_final, scaler, sel_idx, sel_names,
                     n_classes, classes, best_p, acc):
    dump = xgb_final.get_booster().get_dump(dump_format='text')
    n_total_trees = len(dump)
    trees_per_class = n_total_trees // n_classes if n_classes > 1 else n_total_trees

    mean_sel  = scaler.mean_[sel_idx].astype(np.float32)
    scale_sel = scaler.scale_[sel_idx].astype(np.float32)

    L = []
    L.append('/**')
    L.append(' * model.h — XGBoost optimizado para STM32')
    L.append(f' * GridSearchCV params: {best_p}')
    L.append(f' * Accuracy: {acc*100:.2f}%')
    L.append(f' * {n_total_trees} árboles, {len(sel_names)} features')
    L.append(f' * Clases: {dict(enumerate(classes))}')
    L.append(' * Generado por xgboost.py')
    L.append(' */')
    L.append('')
    L.append('#ifndef MODEL_H')
    L.append('#define MODEL_H')
    L.append('')
    L.append(f'#define N_FEATURES     {len(sel_names)}U')
    L.append(f'#define N_CLASSES      {n_classes}U')
    L.append(f'#define N_TREES_TOTAL  {n_total_trees}U')
    L.append(f'#define N_TREES_CLASS  {trees_per_class}U')
    L.append('')
    for i, cls in enumerate(classes):
        safe = str(cls).replace('-', '_').replace(' ', '_').upper()
        L.append(f'#define CLASS_{safe}  {i}')
    L.append('')
    L.append(f'static const int SEL_IDX[N_FEATURES] = {{{", ".join(str(i) for i in sel_idx)}}};')
    L.append('')
    for i, name in enumerate(sel_names):
        L.append(f'#define MEAN_{name:<22} {mean_sel[i]:.8e}f')
    L.append('')
    for i, name in enumerate(sel_names):
        L.append(f'#define SCALE_{name:<21} {scale_sel[i]:.8e}f')
    L.append('')
    norm_body = '\n'.join(
        f'    xn[{i}] = (x_all[SEL_IDX[{i}]] - MEAN_{n}) / SCALE_{n};'
        for i, n in enumerate(sel_names)
    )
    L.append('static inline void normalize_select(const float *x_all, float *xn) {')
    L.append(norm_body)
    L.append('}')
    L.append('')
    for i, tree_str in enumerate(dump):
        L.append(_parse_tree(tree_str, sel_names, i))
        L.append('')

    if n_classes == 2:
        L.append('static inline int predict(const float *x_all) {')
        L.append('    float xn[N_FEATURES];')
        L.append('    normalize_select(x_all, xn);')
        L.append('    float score = 0.0f;')
        for i in range(n_total_trees):
            L.append(f'    score += xgb_tree_{i}(xn);')
        L.append('    float prob = 1.0f / (1.0f + expf(-score));')
        L.append('    return (prob >= 0.5f) ? 1 : 0;')
        L.append('}')
    else:
        L.append('static inline int predict(const float *x_all) {')
        L.append('    float xn[N_FEATURES];')
        L.append('    normalize_select(x_all, xn);')
        L.append(f'    float scores[{n_classes}] = {{0}};')
        for i in range(n_total_trees):
            L.append(f'    scores[{i % n_classes}] += xgb_tree_{i}(xn);')
        L.append('    int best = 0;')
        L.append(f'    for (int i = 1; i < {n_classes}; i++)')
        L.append('        if (scores[i] > scores[best]) best = i;')
        L.append('    return best;')
        L.append('}')

    L.append('')
    L.append('#endif /* MODEL_H */')
    return '\n'.join(L)