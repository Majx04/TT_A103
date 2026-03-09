import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def train_my_model(file_path):
    data = pd.read_csv(file_path)
    label_map = {"flexion": 0, "extension": 1}
    data["label"] = data["etiqueta"].map(label_map)

    # Reutilizamos tu función create_windows_by_sample
    WINDOW_SIZE = 200
    STEP_SIZE = 50

    def create_windows(df, ws, step):
        X, y = [], []
        for sample_id in df["num_muestra"].unique():
            sample = df[df["num_muestra"] == sample_id]
            signal = sample["voltaje"].values
            label = sample["label"].iloc[0]
            for i in range(0, len(signal) - ws, step):
                X.append(signal[i:i + ws])
                y.append(label)
        return np.array(X), np.array(y)

    X, y = create_windows(data, WINDOW_SIZE, STEP_SIZE)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).reshape(X.shape[0], WINDOW_SIZE, 1)
    y = to_categorical(y, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        LSTM(128, return_sequences=True),  # Simplificado para rapidez en web
        Dropout(0.3),
        LSTM(64),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenamos (puedes bajar las épocas para pruebas rápidas)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # Retornamos métricas finales y la historia para gráficas
    return {
        "accuracy": round(acc * 100, 2),
        "loss": round(loss, 4),
        "history": history.history
    }