import pandas as pd

class Preprocessor:
    def __init__(self):
        self.serie = None

    def cargar_serie(self, path, start=None, end=None) -> pd.DataFrame:
        """Carga una serie mensual desde archivo parquet."""
        df = pd.read_parquet(path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.sort_values('fecha')
        df.set_index('fecha', inplace=True)

        df['total'] = df['total_diario'].astype(float) / 1e6  # Normaliza en millones
        df = df[['total']]
        df = df.resample('MS').sum().astype(float)  # Serie mensual
        self.serie = df
        if start is not None: 
            self.serie = df.loc[start:]
        if end is not None:
            self.serie = df.loc[:end] 
        return self.serie

    def tratar_atipicos_por_mes(self) -> pd.DataFrame:
        """Detecta y reemplaza valores atípicos con la mediana mensual."""
        if self.serie is None:
            raise ValueError("Primero debes cargar la serie con `cargar_serie()`.")

        serie = self.serie.copy()
        serie['mes'] = serie.index.month

        for mes in range(1, 13):
            valores_mes = serie[serie['mes'] == mes]['total']
            Q1 = valores_mes.quantile(0.25)
            Q3 = valores_mes.quantile(0.75)
            IQR = Q3 - Q1
            lim_inf = Q1 - 1.125 * IQR
            lim_sup = Q3 + 1.125 * IQR

            outliers = (serie['mes'] == mes) & ((serie['total'] < lim_inf) | (serie['total'] > lim_sup))
            mediana_mes = valores_mes.median()
            serie.loc[outliers, 'total'] = mediana_mes

        self.serie = serie.drop(columns='mes')
        return self.serie
