#model_lstm.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

class LSTMModel:
    def __init__(self, model_path: str, scaler_path: str, input_length: int = 12, output_length: int = 6):
        """Carga el modelo LSTM y el escalador.
        
        Parámetros:
        - model_path: ruta al archivo .keras
        - scaler_path: ruta al archivo .pkl del escalador
        - input_length: longitud de entrada del modelo
        - output_length: cantidad de predicciones futuras
        """
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.input_length = input_length
        self.output_length = output_length

    def predecir(self, serie: pd.DataFrame, fecha_limite: str = None) -> pd.DataFrame:
        """
        Realiza una predicción multistep usando el modelo cargado.

        Parámetros:
        - serie: DataFrame con la serie tratada y su índice como fechas.
        - fecha_limite: si se especifica, acota la serie hasta esa fecha.

        Retorna:
        - DataFrame con fechas futuras y predicción desescalada.
        """
        if fecha_limite:
            serie = serie[serie.index <= fecha_limite]

        # Tomar la última ventana
        ultima_ventana = serie[-self.input_length:].values.reshape(1, self.input_length, 1)
        ventana_escalada = self.scaler.transform(ultima_ventana.reshape(-1, 1)).reshape(1, self.input_length, 1)

        # Predecir
        pred_escalada = self.model.predict(ventana_escalada, verbose=0)
        pred_dese = self.scaler.inverse_transform(pred_escalada).flatten()

        # Crear fechas futuras
        ultima_fecha = serie.index[-1]
        fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1), 
                                       periods=self.output_length, freq='MS')

        return pd.DataFrame({'prediccion': pred_dese}, index=fechas_futuras)
