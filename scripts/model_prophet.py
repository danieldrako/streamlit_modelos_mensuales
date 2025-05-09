# model_prophet.py

import pandas as pd
from prophet import Prophet

class ProphetModel:
    def __init__(self, changepoints: int = 25, changepoint_prior: float = 1.25):
        """Inicializa el modelo Prophet con configuración personalizada."""
        self.model = Prophet(
            n_changepoints=changepoints,
            changepoint_prior_scale=changepoint_prior,
            seasonality_mode='multiplicative',
            interval_width=0.95
        )
        self.model.add_seasonality(name='diciembre', period=365.25, fourier_order=3)

    def entrenar(self, df: pd.DataFrame) -> None:
        """Entrena el modelo Prophet. El DataFrame debe tener columnas: ds (fecha) y y (valor)."""
        self.model.fit(df)

    def predecir(self, n_periodos: int, freq: str = 'MS') -> pd.DataFrame:
        """Genera predicciones futuras."""
        future = self.model.make_future_dataframe(periods=n_periodos, freq=freq)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
    
    def predecir_acotado(self,forecast, fecha_limite: str = None) -> pd.DataFrame:
        forecast[forecast['ds'] >= '2024-12-01']
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')

    def get_model(self) -> Prophet:
        """Devuelve el modelo Prophet crudo (útil para visualizaciones internas)."""
        return self.model
