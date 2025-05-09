#streamlit_app.py
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scripts.preprocessing import Preprocessor
from scripts.model_lstm import *
from scripts.model_sarima import *
from scripts.model_prophet import *

st.set_page_config(
    page_title="Comparador de modelos",
    layout="wide",  # Expande al máximo ancho
    initial_sidebar_state="expanded"
)

# CSS opcional para modo compacto
st.markdown("""
<style>
    .css-18e3th9 { padding: 1rem; }  /* reduce padding en contenido */
</style>
""", unsafe_allow_html=True)

# --- Título
st.title("🔢 Comparador de modelos de predicción mensual")

# --- Sidebar: selección múltiple de modelos
st.sidebar.subheader("Modelos a mostrar:")
mostrar_lstm = st.sidebar.checkbox("Mostrar LSTM", value=True)
mostrar_sarima = st.sidebar.checkbox("Mostrar SARIMA", value=True)
mostrar_prophet = st.sidebar.checkbox("Mostrar Prophet", value=True)

st.sidebar.write("---")
start = "2018-01-01"
end = "2024-11-01"
fecha_limite_max ='2024-11-01'
# --- Preprocesamiento
pre = Preprocessor()
serie = pre.cargar_serie("models/status3totales2017", start=start)
serie_tratada = pre.tratar_atipicos_por_mes()

# Control de fecha en el sidebar
fecha_min = serie_tratada.index.min().to_pydatetime()
fecha_max = (serie_tratada.index.max() - pd.DateOffset(months=12)).to_pydatetime()
fecha_default = fecha_min

fecha_inicio_comparacion = st.sidebar.slider(
    "Selecciona la fecha desde la que deseas visualizar las predicciones:",
    min_value=fecha_min,
    max_value=fecha_max,
    value=fecha_default,
    format="YYYY-MM"
)

# Gráfico 1: Serie original vs tratada
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=serie.index, y=serie['total'],
    mode='lines',
    name='Serie original',
    line=dict(color='rgba(72, 247, 249, 0.7)')
))
fig.add_trace(go.Scatter(
    x=serie_tratada.index, y=serie_tratada['total'],
    mode='lines',
    name='Serie tratada',
    line=dict(color='rgba(113, 234, 7, 0.7)')
))
fig.update_layout(
    title="📈 Serie mensual (tratada vs original)",
    xaxis_title="Fecha",
    yaxis_title="Total mensual",
    hovermode="x unified",
    height=400
)
st.plotly_chart(fig, use_container_width=True, key="grafico_serie_original_tratada")

# Gráfico 2: Comparación de modelos
fig_modelos = go.Figure()
serie_filtrada = serie_tratada[serie_tratada.index >= pd.to_datetime(fecha_inicio_comparacion)]
fig_modelos.add_trace(go.Scatter(
    x=serie_filtrada.index,
    y=serie_filtrada['total'],
    mode='lines',
    name='Serie tratada',
    line=dict(color='rgba(113, 234, 7, 0.7)', dash='dot')
))
fig_modelos.add_trace(go.Scatter(
    x=serie[serie.index >= pd.to_datetime(fecha_inicio_comparacion)].index,
    y=serie[serie.index >= pd.to_datetime(fecha_inicio_comparacion)]['total'],
    mode='lines',
    name='Serie original',
    line=dict(color='rgb(136, 115, 108)', dash='dot')
))

# SARIMA
if mostrar_sarima:
    modelo_sarima = SarimaModel(serie=serie_tratada, fecha_limite=fecha_limite_max)
    #start_pred = fecha_limite_max + pd.DateOffset(months=1)
    sarima_pred = modelo_sarima.predecir(n_periodos=6)
    conf_sarima = modelo_sarima.intervalo_confianza(n_periodos=6)
    print(sarima_pred)

    fig_modelos.add_trace(go.Scatter(
        x=sarima_pred.index,
        y=sarima_pred['prediccion'],
        name="Predicción SARIMA",
        mode="lines+markers",
        line=dict(color='orange', dash='dash')
    ))
    fig_modelos.add_trace(go.Scatter(
        x=conf_sarima.index.tolist() + conf_sarima.index[::-1].tolist(),
        y=conf_sarima.iloc[:, 0].tolist() + conf_sarima.iloc[:, 1][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Confianza SARIMA"
    ))
# LSTM
if mostrar_lstm:
    modelo_lstm = LSTMModel(
        model_path="models/modelo_multistep_mensual.keras",
        scaler_path="models/multistep_mensual_scaler_total.pkl",
        input_length=12,
        output_length=6
    )
    pred_lstm = modelo_lstm.predecir(serie_tratada, fecha_limite=fecha_limite_max)
    fig_modelos.add_trace(go.Scatter(
        x=pred_lstm.index,
        y=pred_lstm['prediccion'],
        name="Predicción LSTM",
        mode="lines+markers",
        line=dict(color='rgb(224, 245, 49)', dash='dot')
    ))

fig_modelos.update_layout(
    title="📊 Comparación de predicciones por modelo",
    xaxis_title="Fecha",
    yaxis_title="Total mensual",
    hovermode="x unified",
    height=550
)
st.plotly_chart(fig_modelos, use_container_width=True, key="grafico_comparacion_modelos")
