#model_sarima.py
import pandas as pd
import numpy as np
import joblib
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

class SarimaModel:
    def __init__(self, model_path: str = None, serie: pd.Series = None,
                 order=(0, 1, 2), seasonal_order=(1, 1, 1, 12), fecha_limite: str = None):
        """
        Si se pasa un model_path, carga un modelo previamente entrenado.
        Si se pasa una serie, entrena un nuevo modelo SARIMA con esa serie.
        """
        if model_path:
            with open(model_path, 'rb') as f:
                self.model: SARIMAXResults = pickle.load(f)
        elif serie is not None:
            if fecha_limite:
                serie = serie[serie.index <= fecha_limite]
            self.ultima_fecha = serie.index[-1]  # <-- 🔧 Aquí se define
            modelo_sarima = SARIMAX(
                serie,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model = modelo_sarima.fit(disp=False)
        else:
            raise ValueError("Debes proporcionar un model_path o una serie para entrenamiento.")

    def guardar(self, output_path: str):
        """Guarda el modelo entrenado en un archivo .pkl"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)

    # def predecir(self, n_periodos: int, start_date: pd.Timestamp, freq: str = "MS") -> pd.DataFrame:
    #     pred = self.model.get_forecast(steps=n_periodos)
    #     pred_values = pred.predicted_mean
    #     pred_index = pd.date_range(start=start_date, periods=n_periodos, freq=freq)
    #     return pd.DataFrame({'prediccion': pred_values}, index=pred_index)
    def predecir(self, n_periodos: int, start_date: pd.Timestamp = None, freq: str = "MS") -> pd.DataFrame:
        pred = self.model.get_forecast(steps=n_periodos)
        pred_values = pred.predicted_mean

        if start_date is None:
            if self.ultima_fecha is None:
                raise ValueError("No se puede inferir start_date: proporciónalo manualmente.")
            start_date = self.ultima_fecha + pd.DateOffset(months=1)

        fechas = pd.date_range(start=start_date, periods=n_periodos, freq=freq)
        return pd.DataFrame({'prediccion': pred_values.values}, index=fechas)

    def intervalo_confianza(self, n_periodos: int, start_date: pd.Timestamp = None, freq: str = "MS") -> pd.DataFrame:
        pred = self.model.get_forecast(steps=n_periodos)
        conf_int = pred.conf_int()

        if start_date is None:
            if self.ultima_fecha is None:
                raise ValueError("No se puede inferir start_date para el intervalo.")
            start_date = self.ultima_fecha + pd.DateOffset(months=1)

        fechas = pd.date_range(start=start_date, periods=n_periodos, freq=freq)
        conf_int.index = fechas
        return conf_int
