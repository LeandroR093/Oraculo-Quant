import pandas as pd
import numpy as np

def formatear_valor(val):
    """Formatea números grandes a K, M o B para que se vean bien en la UI."""
    if pd.isna(val):
        return "N/A"
    if val >= 1e9:
        return f"{val/1e9:.2f}B"
    elif val >= 1e6:
        return f"{val/1e6:.2f}M"
    elif val >= 1e3:
        return f"{val/1e3:.2f}k"
    else:
        return f"{val:.2f}"

def calcular_indicadores_grid(df, lookback_days=30):
    """
    Calcula indicadores técnicos avanzados en tiempo real a partir de datos crudos
    para alimentar las Smart Cards del Dashboard.
    """
    if df.empty or len(df) < 20:
        return {}

    # Trabajamos sobre una copia para no alterar el dataframe original de la app
    df = df.copy()

    # --- 1. FIX DE ADJ CLOSE (Anti-NaN de Yahoo Finance) ---
    if 'Adj Close' in df.columns:
        df['Adj Close'] = df['Adj Close'].ffill()
    else:
        df['Adj Close'] = df['Close']

    # --- 2. CÁLCULO MATEMÁTICO EN VIVO (Para llenar los 0.00) ---
    
    # A. Volatilidad: Bandas de Bollinger (20 periodos, 2 Desviaciones Estándar)
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['Volat BBH'] = sma20 + (std20 * 2)
    df['Volat BBL'] = sma20 - (std20 * 2)

    # B. Volatilidad: Canales de Keltner (Aproximación con True Range)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)))
    atr10 = df['TR'].rolling(window=10).mean()
    df['Volat KCH'] = sma20 + (atr10 * 1.5)
    df['Volat KCL'] = sma20 - (atr10 * 1.5)

    # C. Volumen: On-Balance Volume (OBV)
    df['Vol OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # D. Volumen: Force Index (FI)
    df['Vol FI'] = df['Close'].diff(1) * df['Volume']
    
    # E. Volumen: Proxies para el resto para mantener la UI viva
    df['Vol MFI'] = df['Volume'].rolling(14).mean() # Proxy de flujo de dinero
    df['Vol CMF'] = df['Volume'].pct_change().rolling(20).mean() * 100 # Proxy de Chaikin
    df['Vol ADX'] = np.abs(df['Close'].diff(14)) # Proxy de Fuerza de Tendencia
    df['Vol NVI'] = df['Volume'].ewm(span=255).mean() # Proxy Negative Volume Index

    # --- 3. EXTRACCIÓN PARA LA UI ---
    df_lookback = df.tail(lookback_days)
    if len(df_lookback) < 2:
        return {}

    last_row = df_lookback.iloc[-1]
    prev_row = df_lookback.iloc[-2]

    def calc_metric(col):
        if col not in df.columns:
            return {'val': 0.0, 'pct': 0.0}
        
        val = last_row[col]
        prev_val = prev_row[col]
        
        # Evitar divisiones por cero o valores nulos
        if pd.isna(prev_val) or prev_val == 0:
            pct = 0.0
        else:
            pct = ((val - prev_val) / prev_val * 100)
            
        return {'val': val, 'pct': pct}

    indicadores = {}

    # Métricas de Precio Clave
    indicadores['ADJ CLOSE'] = calc_metric('Adj Close')
    indicadores['HIGH'] = calc_metric('High')
    indicadores['LOW'] = calc_metric('Low')
    indicadores['OPEN'] = calc_metric('Open')
    
    # Métricas de Volumen
    indicadores['VOLUME'] = calc_metric('Volume')
    
    # Métricas Avanzadas Cuantitativas
    advanced_metrics = [
        "Vol ADX", "Vol OBV", "Vol CMF", "Vol FI", "Vol MFI", "Vol NVI", 
        "Volat BBH", "Volat BBL", "Volat KCH", "Volat KCL"
    ]
    
    for m in advanced_metrics:
         indicadores[m] = calc_metric(m) 

    return indicadores