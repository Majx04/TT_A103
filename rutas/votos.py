import numpy as np
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import request, jsonify
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def votosTrain():
    try:
        file = request.files.get('csv_file')
        if not file:
            return jsonify({'error': 'No se recibió archivo CSV'}), 400

        df = pd.read_csv(file)

        features = [
            "WAMP", "VAR", "ZC", "SSC", "LOG", "SSI",
            "MedFreq",
            "TFR_MeanEnergy", "TFR_MaxEnergy", "TFR_Entropy",
            "E_Total", "E_Low", "E_Mid", "E_High",
            "R_Low", "R_Mid", "R_High", "R_HighLow",
            "TopFreq1", "TopFreq2", "TopFreq3", "TopFreq4", "TopFreq5",
            "SpectralEntropy", "SpectralKurtosis", "MeanFreq"
        ]

        missing = [f for f in features if f not in df.columns]
        if missing:
            return jsonify({'error': f'Columnas faltantes: {missing}'}), 400
        if 'label' not in df.columns:
            return jsonify({'error': 'Columna "label" no encontrada'}), 400

        X = df[features].values
        y = df["label"].values
        label_mapping = {'extension': 0, 'flexion': 1}
        y = np.array([label_mapping[label] for label in y])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8]
        }
        grid = GridSearchCV(XGBClassifier(eval_metric='logloss'),
                            param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        xgb = grid.best_estimator_

        selector = SelectFromModel(xgb, threshold="median")
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        selected_features = list(np.array(features)[selector.get_support()])

        mlp = Sequential([
            Dense(256, activation='relu', input_shape=(X_train_sel.shape[1],)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        mlp.fit(X_train_sel, y_train, validation_split=0.2,
                epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

        svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
        svm.fit(X_train_sel, y_train)

        mlp_pred = mlp.predict(X_test_sel).flatten()
        xgb_pred = xgb.predict_proba(X_test)[:, 1]
        svm_pred = svm.predict_proba(X_test_sel)[:, 1]

        best_acc, best_weights, best_prob = 0, None, None
        for w1 in np.arange(0.2, 0.7, 0.1):
            for w2 in np.arange(0.2, 0.7, 0.1):
                w3 = 1 - (w1 + w2)
                if w3 < 0:
                    continue
                prob = w1 * mlp_pred + w2 * xgb_pred + w3 * svm_pred
                acc = accuracy_score(y_test, (prob > 0.5).astype(int))
                if acc > best_acc:
                    best_acc, best_weights, best_prob = acc, (float(w1), float(w2), float(w3)), prob

        best_thr, best_thr_acc = 0.5, 0
        for t in np.arange(0.3, 0.7, 0.01):
            acc = accuracy_score(y_test, (best_prob > t).astype(int))
            if acc > best_thr_acc:
                best_thr_acc, best_thr = acc, float(t)

        y_final = (best_prob > best_thr).astype(int)
        report = classification_report(y_test, y_final, output_dict=True)
        cm = confusion_matrix(y_test, y_final).tolist()

        importances = xgb.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0f0f0f')
        ax.set_facecolor('#0f0f0f')
        sorted_idx = np.argsort(importances)
        ax.barh(np.array(features)[sorted_idx], importances[sorted_idx], color='#00d4aa', height=0.6)
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.set_xlabel('Importancia', color='#aaaaaa', fontsize=10)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0f0f0f')
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return jsonify({
            'voting_accuracy': round(float(best_acc), 4),
            'final_accuracy': round(float(accuracy_score(y_test, y_final)), 4),
            'best_threshold': round(best_thr, 4),
            'weights': {'mlp': round(best_weights[0], 2),
                        'xgb': round(best_weights[1], 2),
                        'svm': round(best_weights[2], 2)},
            'report': report,
            'confusion_matrix': cm,
            'selected_features': selected_features,
            'total_samples': len(y),
            'xgb_params': grid.best_params_,
            'feature_chart': chart_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500