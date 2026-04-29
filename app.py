import streamlit as st
import pandas as pd
import pickle
import glob
import yfinance as yf
import ta
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from indicators import calcular_indicadores_grid, formatear_valor
from simulation import ejecutar_monte_carlo

st.set_page_config(page_title="Advanced Quant Trading Platform", page_icon="📈", layout="wide")

# --- CUSTOM CSS INJECTION ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; color: #f0f6fc; }
    .main { background-color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; padding-top: 2rem; }
    .smart-card {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px;
        display: flex; flex-direction: column; justify-content: space-between; min-height: 160px;
        transition: transform 0.2s ease, border-color 0.2s ease; margin-bottom: 16px;
    }
    .smart-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
    .card-title { color: #8b949e; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
    .card-value { font-size: 1.6rem; font-weight: 700; color: #f0f6fc; font-family: 'JetBrains Mono', monospace; }
    .pct-up { color: #3fb950; font-weight: bold; }
    .pct-down { color: #f85149; font-weight: bold; }
    .macro-alert { padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid; }
</style>
""", unsafe_allow_html=True)

# --- RUTAS INTELIGENTES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir) if os.path.basename(current_dir) == "src" else current_dir
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'outputs', 'oraculo_xgboost.pkl')

@st.cache_resource
def cargar_oraculo():
    try:
        with open(MODEL_PATH, 'rb') as f:
            modelo = pickle.load(f)
    except Exception:
        modelo = None
    archivos_parquet = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    tickers = sorted([os.path.basename(f).replace('.parquet', '') for f in archivos_parquet])
    return modelo, tickers

modelo_ml, lista_tickers = cargar_oraculo()

@st.cache_data(ttl=3600)
def descargar_datos(ticker):
    filepath = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        if 'Date' not in df.columns and 'index' in df.columns:
            df = df.rename(columns={'index': 'Date'})
        elif 'Date' not in df.columns:
            df = df.reset_index()
            if 'index' in df.columns: df = df.rename(columns={'index': 'Date'})
            
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
            df.set_index('Date', inplace=True)
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
        return df
    return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("Selección de Activos")
    default_tk = ['SPY'] if 'SPY' in lista_tickers else [lista_tickers[0]] if lista_tickers else []
    acciones_seleccionadas = st.multiselect(
        "Comparar Acciones", 
        options=lista_tickers, 
        default=default_tk, 
        max_selections=10,
        help="Selecciona una o varias empresas para analizar."
    )
    
    st.markdown("---")
    st.subheader("Proyección Quant")
    # FIX: Aclaración de horizontes
    st.info("🎯 **Monte Carlo:** Riesgo a 30 Días\n\n🤖 **Oráculo IA:** Operación a 1 Día (Mañana)")
    horizonte_dias = 30
    
    st.markdown("---")
    st.subheader("Visualización")
    tipo_grafico = st.selectbox("Tipo de Visualización", options=["Línea", "Velas", "Área", "Barras"], index=0)
    mostrar_volumen = st.checkbox("Mostrar Volumen", value=True)
    mostrar_mc_paths = st.checkbox("Ver Rutas Proyectadas", value=True)
    mostrar_grid = st.checkbox("Ver Parámetros Clave", value=True)
    
    st.markdown("---")
    st.subheader("Gestión de Riesgo")
    vol_mult = st.slider("Multiplicador de Volatilidad", 1.0, 3.0, 1.0, step=0.1)

st.title("Advanced Quant Trading Platform")
period = st.radio("Período de Visualización", ["1M", "3M", "6M", "YTD", "1A", "MAX"], index=4, horizontal=True)

if not acciones_seleccionadas:
    st.warning("Selecciona al menos una acción.")
    st.stop()

@st.cache_data
def obtener_simulacion_cache(df_c, d_proyeccion, v_mult):
    if not df_c.empty and len(df_c) > 20:
        return ejecutar_monte_carlo(df_c, dias_proyeccion=d_proyeccion, n_simulaciones=1000, vol_mult=v_mult)
    return pd.DataFrame(), {}

datos_dict, sim_dict, stats_dict, ml_preds = {}, {}, {}, {}

@st.cache_data(ttl=3600)
def obtener_macro_vivos():
    try:
        macro_tickers = ['SPY', '^TNX', '^VIX']
        raw = yf.download(macro_tickers, period='5d', progress=False)['Close']
        m = pd.DataFrame(index=raw.index)
        m['SPY_Ret'] = raw['SPY'].pct_change()
        m['TNX_Val'] = raw['^TNX']
        m['TNX_Cambio'] = raw['^TNX'].diff()
        m['VIX_Val'] = raw['^VIX']
        m['VIX_Cambio'] = raw['^VIX'].diff()
        return m.iloc[-1:]
    except:
        return pd.DataFrame()

macro_now = obtener_macro_vivos()

with st.spinner("Sincronizando Radar Macroeconómico e IA..."):
    for ticker in acciones_seleccionadas:
        df = descargar_datos(ticker)
        if not df.empty and len(df) > 30:
            df_s, stats_s = obtener_simulacion_cache(df, horizonte_dias, vol_mult)
            datos_dict[ticker] = df
            sim_dict[ticker] = df_s
            stats_dict[ticker] = stats_s
            
            if modelo_ml is not None:
                df_ml = df.copy()
                df_ml['Retorno_Diario'] = df_ml['Close'].pct_change()
                df_ml['RSI'] = ta.momentum.rsi(df_ml['Close'], window=14)
                
                bb_u = ta.volatility.bollinger_hband(df_ml['Close'], window=20, window_dev=2)
                bb_l = ta.volatility.bollinger_lband(df_ml['Close'], window=20, window_dev=2)
                df_ml['BB_Posicion'] = (df_ml['Close'] - bb_l) / (bb_u - bb_l)
                
                df_ml['MACD_Hist'] = ta.trend.macd_diff(df_ml['Close'], window_fast=12, window_slow=26, window_sign=9)
                
                sma20 = ta.trend.sma_indicator(df_ml['Close'], window=20)
                sma50 = ta.trend.sma_indicator(df_ml['Close'], window=50)
                sma200 = ta.trend.sma_indicator(df_ml['Close'], window=200)
                df_ml['Dist_SMA20'] = (df_ml['Close'] / sma20) - 1
                df_ml['Dist_SMA50'] = (df_ml['Close'] / sma50) - 1
                df_ml['Dist_SMA200'] = (df_ml['Close'] / sma200) - 1
                
                df_ml['Volatilidad_5d'] = df_ml['Retorno_Diario'].rolling(window=5).std()
                
                if not macro_now.empty and 'VIX_Val' in macro_now.columns:
                    vix_actual = macro_now['VIX_Val'].values[0]
                    vix_hist_mean = df_ml['Close'].rolling(20).mean().iloc[-1] # Proxy si no hay VIX en hist
                    df_ml['VIX_Tendencia'] = (vix_actual / vix_hist_mean) - 1 if vix_hist_mean != 0 else 0

                df_ml['Retorno_Ayer'] = df_ml['Retorno_Diario'].shift(1)
                df_ml['Retorno_Anteayer'] = df_ml['Retorno_Diario'].shift(2)
                df_ml['RSI_Ayer'] = df_ml['RSI'].shift(1)
                df_ml['MACD_Hist_Ayer'] = df_ml['MACD_Hist'].shift(1)
                
                last_row = df_ml.iloc[-1:].copy()
                last_row['Ticker'] = ticker
                
                if not macro_now.empty:
                    for col in macro_now.columns:
                        last_row[col] = macro_now[col].values[0]
                
                last_row['Ticker'] = last_row['Ticker'].astype('category')
                cols_needed = modelo_ml.feature_names_in_
                X_pred = last_row.reindex(columns=cols_needed, fill_value=0)
                
                try:
                    pred = modelo_ml.predict(X_pred)[0]
                    proba = modelo_ml.predict_proba(X_pred)[0][pred]
                    
                    # FIX: Traducción de la IA de Alto Rendimiento
                    if pred == 1:
                        texto_ia = "🔥 SALTO ALCISTA (>0.25%)"
                        color_ia = "#3fb950"
                    else:
                        texto_ia = "⚠️ RIESGO / PLANO"
                        color_ia = "#d29922" # Naranja de prevención
                        
                    ml_preds[ticker] = {"texto": texto_ia, "confianza": proba * 100, "color": color_ia}
                except Exception as e:
                    ml_preds[ticker] = {"texto": f"ERR", "confianza": 0.0, "color": "#8b949e"}

st.markdown("---")

# 1. TERMÓMETRO MACROECONÓMICO (NUEVO)
if not macro_now.empty and 'VIX_Val' in macro_now.columns:
    vix_val = macro_now['VIX_Val'].values[0]
    if vix_val > 25:
        bg_color, border_color = "rgba(248, 81, 73, 0.1)", "#f85149"
        vix_status = f"🔴 MODO PÁNICO | VIX: {vix_val:.2f}"
        vix_msg = "Las instituciones están cubriendo carteras por miedo extremo. <b>Recomendación Quant:</b> Mantente en efectivo o reduce el tamaño de tus posiciones al mínimo."
    elif vix_val > 15:
        bg_color, border_color = "rgba(210, 153, 34, 0.1)", "#d29922"
        vix_status = f"🟡 MODO CAUTELA | VIX: {vix_val:.2f}"
        vix_msg = "Mercado nervioso e indeciso. <b>Recomendación Quant:</b> Busca operaciones rápidas tipo 'Hit and Run' guiadas por el Oráculo, no te quedes expuesto muchos días."
    else:
        bg_color, border_color = "rgba(63, 185, 80, 0.1)", "#3fb950"
        vix_status = f"🟢 MODO SEGURO | VIX: {vix_val:.2f}"
        vix_msg = "Miedo bajo e institucionalidad comprando. <b>Recomendación Quant:</b> Entorno ideal para seguir las predicciones del Oráculo con confianza normal."
        
    st.markdown(f"""
    <div class="macro-alert" style="background-color: {bg_color}; border-color: {border_color};">
        <h4 style="margin-top: 0; color: {border_color};">{vix_status}</h4>
        <span style="font-size: 0.95rem;">{vix_msg}</span>
    </div>
    """, unsafe_allow_html=True)


# 2. Copiloto Inteligente
st.subheader("Copiloto de Inteligencia Financiera (Predicción a 1 Día)")
for i, (tk, stats) in enumerate(stats_dict.items()):
    if i % 4 == 0: cols_insight = st.columns(min(len(stats_dict) - i, 4), gap="medium")
    with cols_insight[i % 4]:
        score = stats.get('score_confianza', 0.0)
        driver = stats.get('driver_principal', 'N/A')
        status = "Alto" if score > 75 else "Medio" if score > 45 else "Bajo"
        color = "#3fb950" if status == "Alto" else "#d29922" if status == "Medio" else "#f85149"
        
        ia_texto = ml_preds[tk]['texto'] if tk in ml_preds else "N/A"
        ia_conf = ml_preds[tk]['confianza'] if tk in ml_preds else 0.0
        ia_color = ml_preds[tk]['color'] if tk in ml_preds else "#8b949e"
        
        st.markdown(f"""
        <div class="smart-card" style="border-left: 4px solid {color}; padding: 16px;">
            <div class="card-title">{tk} | Mando Estocástico</div>
            <div class="card-value" style="color: {color};">{score:.0f}% Risk Conf.</div>
            <div style="font-size: 0.85rem; margin-top: 12px; line-height: 1.6;">
                <b>🤖 IA:</b> <span style="color: {ia_color}; font-weight: bold;">{ia_texto} ({ia_conf:.1f}%)</span><br>
                <b>⚙️ Motor:</b> {driver}<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# 3. Estado de Activos
st.subheader("Estado de Activos")
for i, (tk, df) in enumerate(datos_dict.items()):
    if i % 5 == 0: cols_precio = st.columns(min(len(datos_dict) - i, 5), gap="medium")
    with cols_precio[i % 5]:
        ultimo_cierre = df['Close'].iloc[-1]
        cierre_anterior = df['Close'].iloc[-2]
        cambio_pct = ((ultimo_cierre - cierre_anterior) / cierre_anterior) * 100
        color_class = "pct-up" if cambio_pct > 0 else "pct-down" if cambio_pct < 0 else ""
        volumen_texto = f"<br><span style='font-size: 0.75rem; color: #8b949e;'>Vol: {formatear_valor(df['Volume'].iloc[-1])}</span>" if 'Volume' in df.columns else ""
        
        st.markdown(f"""
        <div class="smart-card" style="min-height: 100px; padding: 16px; text-align: center;">
            <div class="card-title">{tk}</div>
            <div class="card-value" style="font-size: 1.2rem;">${ultimo_cierre:.2f}</div>
            <div class="{color_class}" style="font-size: 0.85rem;">{"+" if cambio_pct > 0 else ""}{cambio_pct:.2f}%</div>
            {volumen_texto}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# 4. Matriz de Correlación
if len(datos_dict) > 1:
    st.subheader("🔗 Matriz de Correlación (Sinergia y Diversificación)")
    st.info("💡 **¿Para qué sirve esto?** Si dos activos tienen un valor cercano a 1.0 (Rojo), se mueven casi igual. Para proteger tu dinero, busca empresas con correlación baja o negativa (Azul) para que si una cae, la otra sirva de escudo.")
    df_combined = pd.DataFrame({tk: df['Close'] for tk, df in datos_dict.items()}).dropna()
    if not df_combined.empty:
        df_corr = df_combined.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=df_corr.values, x=df_corr.columns, y=df_corr.index, colorscale='balance', zmin=-1, zmax=1,
            text=np.round(df_corr.values, 2), texttemplate="%{text}"
        ))
        fig_corr.update_layout(template="plotly_dark", height=280, margin=dict(t=20, b=20, l=40, r=40))
        st.plotly_chart(fig_corr, use_container_width=True)

# 5. Unified Canvas
st.subheader("Evolución y Rendimiento Proyectado")

is_comparison = len(datos_dict) > 1

if mostrar_volumen and not is_comparison:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
else:
    fig = go.Figure()

colores = ['#58a6ff', '#3fb950', '#f85149', '#d29922', '#bc8cff', '#ffa657']

for i, (tk, df_data) in enumerate(datos_dict.items()):
    color = colores[i % len(colores)]
    df_sim = sim_dict.get(tk, pd.DataFrame())
    
    dias_filtro = {"1M": 30, "3M": 90, "6M": 180, "YTD": 120, "1A": 252, "MAX": len(df_data)}.get(period, 30)
    df_plot = df_data.tail(dias_filtro)
    if df_plot.empty: continue
    
    ultima_fecha = df_plot.index[-1]
    frecuencia_tiempo = 'D' if "-" in tk else 'B'
    fechas_completas_sim = pd.date_range(start=ultima_fecha, periods=horizonte_dias + 1, freq=frecuencia_tiempo)

    if not is_comparison:
        if tipo_grafico == "Velas":
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name=f'{tk} Histórico'), row=1 if mostrar_volumen else None, col=1 if mostrar_volumen else None)
        elif tipo_grafico == "Barras":
            fig.add_trace(go.Ohlc(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name=f'{tk} Histórico'), row=1 if mostrar_volumen else None, col=1 if mostrar_volumen else None)
        else:
            fill = 'tozeroy' if tipo_grafico == "Área" else None
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], mode='lines', name=f'{tk} Histórico', line=dict(color=color, width=2), fill=fill), row=1 if mostrar_volumen else None, col=1 if mostrar_volumen else None)
        
        if mostrar_volumen and 'Volume' in df_plot.columns:
            vol_colors = ['#f85149' if row['Close'] < row['Open'] else '#3fb950' for index, row in df_plot.iterrows()]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], name='Volumen', marker_color=vol_colors, opacity=0.8), row=2, col=1)

        if not df_sim.empty and mostrar_mc_paths:
            ultimo_valor_mediana = df_sim.median(axis=1).iloc[-1]
            idx_min = (df_sim.iloc[-1] - ultimo_valor_mediana).abs().idxmin()
            ruta_jagged = df_sim[idx_min] if pd.notna(idx_min) else df_sim.iloc[:, 0]
            
            if tipo_grafico in ["Velas", "Barras"]:
                spread_hist = df_plot['High'].mean() / df_plot['Low'].mean() if not df_plot.empty else 1.005
                vol_diaria = (spread_hist - 1) / 2
                
                proj_close = ruta_jagged.values
                proj_open = np.roll(proj_close, 1)
                proj_open[0] = df_plot['Close'].iloc[-1]
                
                proj_high = np.maximum(proj_open, proj_close) * (1 + vol_diaria)
                proj_low = np.minimum(proj_open, proj_close) * (1 - vol_diaria)
                
                if tipo_grafico == "Velas":
                    fig.add_trace(go.Candlestick(
                        x=fechas_completas_sim, open=proj_open, high=proj_high, low=proj_low, close=proj_close,
                        name=f'{tk} Proyección Velas',
                        increasing_line_color='#FFD700', decreasing_line_color='#B8860B'
                    ), row=1 if mostrar_volumen else None, col=1 if mostrar_volumen else None)
                else:
                    fig.add_trace(go.Ohlc(
                        x=fechas_completas_sim, open=proj_open, high=proj_high, low=proj_low, close=proj_close,
                        name=f'{tk} Proyección Barras',
                        increasing_line_color='#FFD700', decreasing_line_color='#B8860B'
                    ), row=1 if mostrar_volumen else None, col=1 if mostrar_volumen else None)
            else:
                fig.add_trace(go.Scatter(
                    x=fechas_completas_sim, y=ruta_jagged, mode='lines', 
                    name=f'{tk} Proyección', line=dict(color="#FFD700", width=3, dash='dot')
                ), row=1 if mostrar_volumen else None, col=1 if mostrar_volumen else None)
            
            p10_line = df_sim.apply(lambda x: np.percentile(x, 10), axis=1)
            p90_line = df_sim.apply(lambda x: np.percentile(x, 90), axis=1)
            
            fig.add_trace(go.Scatter(x=fechas_completas_sim, y=p90_line, mode='lines', line=dict(color='rgba(63, 185, 80, 0.4)', width=1, dash='dot'), name='Optimista (P90)'), row=1 if mostrar_volumen else None, col=1 if mostrar_volumen else None)
            fig.add_trace(go.Scatter(x=fechas_completas_sim, y=p10_line, mode='lines', line=dict(color='rgba(248, 81, 73, 0.4)', width=1, dash='dot'), name='Pesimista (P10)', fill='tonexty', fillcolor='rgba(88, 166, 255, 0.05)'), row=1 if mostrar_volumen else None, col=1 if mostrar_volumen else None)
    else:
        base_price = df_plot['Close'].iloc[0]
        hist_pct = (df_plot['Close'] / base_price - 1) * 100
        fig.add_trace(go.Scatter(x=df_plot.index, y=hist_pct, mode='lines', name=f'{tk}', line=dict(color=color, width=2)))
        
        if not df_sim.empty and mostrar_mc_paths:
            ultimo_valor_mediana = df_sim.median(axis=1).iloc[-1]
            idx_min = (df_sim.iloc[-1] - ultimo_valor_mediana).abs().idxmin()
            ruta_repr = df_sim[idx_min] if pd.notna(idx_min) else df_sim.iloc[:, 0]
            sim_pct = (ruta_repr / base_price - 1) * 100
            fig.add_trace(go.Scatter(x=fechas_completas_sim, y=sim_pct, mode='lines', name=f'{tk} Proj.', line=dict(color=color, width=2, dash='dot')))

if not is_comparison and not ("-" in acciones_seleccionadas[0]):
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

title_y = "Rendimiento Normalizado (%)" if is_comparison else f"Precio ($)"
fig.update_layout(
    template="plotly_dark", hovermode="x unified", margin=dict(l=40, r=40, t=10, b=40),
    height=650, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
if not is_comparison and mostrar_volumen:
    fig.update_yaxes(title_text=title_y, row=1, col=1, showgrid=True, gridcolor="#21262d")
    fig.update_yaxes(title_text="Vol", row=2, col=1, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#21262d", rangeslider=dict(visible=False))
else:
    fig.update_yaxes(title_text=title_y, showgrid=True, gridcolor="#21262d")
    fig.update_xaxes(showgrid=True, gridcolor="#21262d", rangeslider=dict(visible=False))

st.plotly_chart(fig, use_container_width=True)

# 6. Análisis de Parámetros Clave
if mostrar_grid:
    st.divider()
    st.subheader("Análisis de Parámetros Clave")
    master_indicadores = {}
    for tk_id, df_tk in datos_dict.items():
        dias_hist = {"1M": 30, "3M": 90, "6M": 180, "YTD": 120, "1A": 252, "MAX": len(df_tk)}.get(period, 30)
        ind_hist = calcular_indicadores_grid(df_tk, lookback_days=dias_hist)
        
        df_sim_tk = sim_dict.get(tk_id, pd.DataFrame())
        ind_fut = {}
        if not df_sim_tk.empty:
            try:
                ultimo_val_med = df_sim_tk.median(axis=1).iloc[-1]
                idx_repr = (df_sim_tk.iloc[-1] - ultimo_val_med).abs().idxmin()
                camino_repr = df_sim_tk[idx_repr].values if pd.notna(idx_repr) else df_sim_tk.iloc[:, 0].values
                
                df_fut_tk = df_tk.copy()
                prom_vol = df_tk['Volume'].mean()
                fechas_f_tk = pd.date_range(start=df_tk.index[-1], periods=horizonte_dias + 1, freq='D' if "-" in tk_id else 'B')
                
                filas_fut = []
                for px in camino_repr[1:]:
                    spread = df_tk['High'].mean() / df_tk['Low'].mean() if not df_tk.empty else 1.005
                    filas_fut.append({'Open': px, 'High': px * spread, 'Low': px / spread, 'Close': px, 'Adj Close': px, 'Volume': prom_vol})
                
                df_append_tk = pd.DataFrame(filas_fut, index=fechas_f_tk[1:])
                df_total_tk = pd.concat([df_fut_tk, df_append_tk])
                ind_fut = calcular_indicadores_grid(df_total_tk, lookback_days=horizonte_dias)
            except: pass
        master_indicadores[tk_id] = {'hist': ind_hist, 'fut': ind_fut}

    categorias = {
        "Precio": ["ADJ CLOSE", "HIGH", "LOW", "OPEN"],
        "Volumen": ["VOLUME", "Vol ADX", "Vol OBV", "Vol CMF", "Vol FI", "Vol MFI", "Vol NVI"],
        "Volatilidad": ["Volat BBH", "Volat BBL", "Volat KCH", "Volat KCL"]
    }

    for cat_nombre, metricas_cat in categorias.items():
        st.markdown(f"#### {cat_nombre}")
        cols_cat = st.columns(4)
        for i, m in enumerate(metricas_cat):
            with cols_cat[i % 4]:
                html_assets = ""
                for tk_id in acciones_seleccionadas:
                    if tk_id in master_indicadores and m in master_indicadores[tk_id]['hist']:
                        data_h = master_indicadores[tk_id]['hist'][m]
                        data_f = master_indicadores[tk_id]['fut'].get(m, {})
                        val_now = data_h['val']
                        pct_p = data_h.get('pct', 0.0)
                        val_f = data_f.get('val', val_now)
                        pct_f = ((val_f - val_now) / val_now * 100) if val_now != 0 else 0
                        
                        c_p = "color: #3fb950;" if pct_p > 0 else "color: #f85149;"
                        c_f = "color: #FFD700;" if pct_f > 0 else "color: #f85149;"
                        bar_w = min(abs(pct_p)*5, 100)
                        bar_c = "#3fb950" if pct_p > 0 else "#f85149"

                        html_assets += f'<div style="padding: 8px 0; border-bottom: 1px solid #30363d;">'
                        html_assets += f'<div style="display: flex; justify-content: space-between; align-items: center;"><span style="font-weight: 700; color: #58a6ff; font-size: 0.85rem;">{tk_id}</span><span style="font-size: 1rem; font-weight: 500;">{formatear_valor(val_now)}</span></div>'
                        html_assets += f'<div style="display: flex; justify-content: space-between; font-size: 0.7rem; margin-top: 2px;"><span style="{c_p}">Hist: {pct_p:+.1f}%</span><span style="{c_f} font-weight: 700;">Proj: {pct_f:+.1f}%</span></div>'
                        html_assets += f'<div style="height: 2px; background: #21262d; border-radius: 1px; margin-top: 4px;"><div style="width: {bar_w}%; height: 100%; background: {bar_c}; border-radius: 1px;"></div></div></div>'
                
                full_card_html = f'<div class="smart-card" style="margin-bottom: 20px; border-top: 2px solid #58a6ff; background: #161b22; border-radius: 6px; padding: 10px; min-height: auto;">'
                full_card_html += f'<div style="font-size: 0.9rem; font-weight: 600; color: #8b949e; margin-bottom: 8px; text-transform: uppercase;">{m}</div>{html_assets}</div>'
                st.markdown(full_card_html, unsafe_allow_html=True)

# 7. Máquina del Tiempo (Backtesting Pedagógico)
st.divider()
st.subheader("⏱️ Máquina del Tiempo (Prueba de Estrategia)")

st.info("""
💡 **¿Por qué probar estrategias?** Los novatos solo miran cuánto dinero ganan. Los profesionales miran **cuánto evitan perder** durante las crisis (Peor Caída).
* **Comprar y Olvidar (Buy & Hold):** Gana más en mercados alcistas, pero te comes todas las crisis financieras de lleno.
* **Estrategia Inteligente:** Sacrifica un poco de ganancia máxima a cambio de detectar las crisis y salirse del mercado a tiempo. **Fíjate en la métrica "Peor Caída"**.
""")

fig_backtest = go.Figure()
colores_bt = ['#3fb950', '#58a6ff', '#f85149', '#d29922', '#bc8cff', '#ffa657']

def calcular_drawdown(cum_returns):
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns / roll_max) - 1.0
    return drawdown.min() * 100

for i, (tk, df_bt) in enumerate(datos_dict.items()):
    color = colores_bt[i % len(colores_bt)]
    df_bt_copy = df_bt.copy()
    
    col_precio = 'Adj Close' if 'Adj Close' in df_bt_copy.columns else 'Close'
    
    df_bt_copy['MA20'] = df_bt_copy[col_precio].rolling(window=20).mean()
    df_bt_copy['MA50'] = df_bt_copy[col_precio].rolling(window=50).mean()
    df_bt_copy['Signal'] = 0
    df_bt_copy.loc[df_bt_copy['MA20'] > df_bt_copy['MA50'], 'Signal'] = 1
    
    df_bt_copy['Retorno_BH'] = df_bt_copy[col_precio].pct_change()
    df_bt_copy['Retorno_Estrategia'] = df_bt_copy['Retorno_BH'] * df_bt_copy['Signal'].shift(1)
    
    rend_cum_est = (1 + df_bt_copy['Retorno_Estrategia']).dropna().cumprod()
    rend_cum_bh = (1 + df_bt_copy['Retorno_BH']).dropna().cumprod()
    
    dd_est = calcular_drawdown(rend_cum_est)
    dd_bh = calcular_drawdown(rend_cum_bh)
    
    fig_backtest.add_trace(go.Scatter(
        x=rend_cum_est.index, y=(rend_cum_est - 1) * 100, 
        name=f"Estrategia {tk} (Peor Caída: {dd_est:.1f}%)", 
        line=dict(color=color, width=2)
    ))
    
    if len(datos_dict) == 1 or i == 0:
        nombre_bh = f"Comprar y Olvidar {tk} (Peor Caída: {dd_bh:.1f}%)"
        fig_backtest.add_trace(go.Scatter(
            x=rend_cum_bh.index, y=(rend_cum_bh - 1) * 100, 
            name=nombre_bh, 
            line=dict(color="#8b949e", dash='dash', width=1 if len(datos_dict) > 1 else 2)
        ))

fig_backtest.update_layout(
    template="plotly_dark", height=450, margin=dict(l=40, r=40, t=20, b=40),
    yaxis=dict(title_text="Rendimiento Acumulado (%)", showgrid=True, gridcolor="#21262d"),
    xaxis=dict(showgrid=True, gridcolor="#21262d"),
    hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_backtest, use_container_width=True)