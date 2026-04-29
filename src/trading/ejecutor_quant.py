import os
import glob
import pickle
import pandas as pd
import ta
import math
import warnings
from ib_insync import *
warnings.filterwarnings('ignore')

# --- RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(ROOT_DIR, 'data', 'outputs', 'oraculo_xgboost.pkl')

# --- CONFIGURACIÓN DE RIESGO Y CARTERA ---
CAPITAL_TOTAL = 2000.0          # Dólares virtuales totales a invertir hoy
TOP_N_ACCIONES = 2              # Número de acciones a comprar (Diversificación)
TAKE_PROFIT_PCT = 1.02          # +2.0% de ganancia
STOP_LOSS_PCT = 0.985           # -1.5% de pérdida

def buscar_mejores_acciones():
    """Analiza todas las acciones y devuelve el Top N con mayor confianza."""
    print("🧠 Consultando al Oráculo para la ejecución diversificada...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            modelo = pickle.load(f)
    except Exception:
        print("❌ Error: No se encontró el modelo XGBoost.")
        return []

    archivos = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    activos_alcistas = []
    
    for f in archivos:
        ticker = os.path.basename(f).replace('.parquet', '')
        try:
            df = pd.read_parquet(f)
            if df.empty or len(df) < 30: continue
            
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
            
            df_ml['Retorno_Ayer'] = df_ml['Retorno_Diario'].shift(1)
            df_ml['Retorno_Anteayer'] = df_ml['Retorno_Diario'].shift(2)
            df_ml['RSI_Ayer'] = df_ml['RSI'].shift(1)
            df_ml['MACD_Hist_Ayer'] = df_ml['MACD_Hist'].shift(1)
            
            df_ml['Volatilidad_5d'] = df_ml['Retorno_Diario'].rolling(window=5).std()
            df_ml['VIX_Tendencia'] = 0 
            
            last_row = df_ml.iloc[-1:].copy()
            last_row['Ticker'] = ticker
            last_row['Ticker'] = last_row['Ticker'].astype('category')
            
            cols_needed = modelo.feature_names_in_
            X_pred = last_row.reindex(columns=cols_needed, fill_value=0)
            
            pred = modelo.predict(X_pred)[0]
            proba = modelo.predict_proba(X_pred)[0][pred]
            
            if pred == 1:
                precio_actual = df['Close'].iloc[-1]
                activos_alcistas.append((ticker, proba * 100, precio_actual))
                
        except Exception: pass
            
    if activos_alcistas:
        # Ordenamos de mayor a menor confianza
        activos_alcistas.sort(key=lambda x: x[1], reverse=True)
        # Devolvemos solo las Top N seleccionadas
        return activos_alcistas[:TOP_N_ACCIONES]
    return []

def ejecutar_ordenes_bracket():
    mejores_acciones = buscar_mejores_acciones()
    
    if not mejores_acciones:
        print("⚠️ El Oráculo no encontró oportunidades claras hoy. Manteniendo efectivo.")
        return

    # Dividimos el capital equitativamente entre las acciones seleccionadas
    capital_por_accion = CAPITAL_TOTAL / len(mejores_acciones)
    
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=3)
        print("✅ Conectado a Interactive Brokers.")
        print(f"💼 Capital total a invertir: ${CAPITAL_TOTAL} USD dividido en {len(mejores_acciones)} activos.\n")
        
        for ticker, confianza, precio_ref in mejores_acciones:
            print(f"🎯 PREPARANDO: {ticker} (Confianza: {confianza:.1f}%) | Ref: ${precio_ref:.2f}")
            
            accion = Stock(ticker, 'SMART', 'USD')
            ib.qualifyContracts(accion)
            
            cantidad_acciones = math.floor(capital_por_accion / precio_ref)
            if cantidad_acciones < 1: 
                print(f"⚠️ Capital insuficiente para comprar 1 acción de {ticker}. Saltando...")
                continue
                
            precio_tp = round(precio_ref * TAKE_PROFIT_PCT, 2)
            precio_sl = round(precio_ref * STOP_LOSS_PCT, 2)
            
            print(f"   -> Comprando {cantidad_acciones} acciones")
            print(f"   -> TP: ${precio_tp} | SL: ${precio_sl}")
            
            # Orden Bracket
            bracket = ib.bracketOrder(
                'BUY', 
                cantidad_acciones, 
                limitPrice=round(precio_ref, 2), 
                takeProfitPrice=precio_tp, 
                stopLossPrice=precio_sl
            )
            
            # Aseguramos entrada a mercado
            bracket.parent.orderType = 'MKT'
            bracket.parent.limitPrice = 0.0
            
            for orden in bracket:
                ib.placeOrder(accion, orden)
                
            print(f"🚀 Órdenes enviadas para {ticker}.\n")
            ib.sleep(1) # Pequeña pausa para no saturar al broker
            
        print("✅ TODAS LAS OPERACIONES FUERON ENVIADAS A WALL STREET.")

    except Exception as e:
        print(f"❌ Error de ejecución en IBKR: {e}")
    finally:
        ib.disconnect()

if __name__ == "__main__":
    ejecutar_ordenes_bracket()