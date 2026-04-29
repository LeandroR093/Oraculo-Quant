import os
import glob
import yfinance as yf
import pandas as pd
import logging
import sys

# ==============================================================================
# CONFIGURACIÓN DEL ACTUALIZADOR AUTÓNOMO
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- RUTAS DINÁMICAS (CRÍTICO PARA IMPORTAR src) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Como el archivo está en 'src', solo subimos UN nivel hasta la raíz del proyecto
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

# Importamos tu clase maestra infalible
from src.features.build_features import FeatureEngineer

PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

def obtener_datos_macro():
    """Descarga el pulso de la economía global"""
    logger.info("Descargando Datos Macroeconómicos e Índice del Miedo (VIX)...")
    tickers_macro = ['SPY', '^TNX', '^VIX']
    
    df_raw = yf.download(tickers_macro, period="max", progress=False)
    
    try: df_macro = df_raw['Close'].copy()
    except KeyError: df_macro = df_raw.copy()

    df_macro.ffill(inplace=True)
    m = pd.DataFrame(index=df_macro.index)
    
    if 'SPY' in df_macro.columns: m['SPY_Ret'] = df_macro['SPY'].pct_change()
    if '^TNX' in df_macro.columns:
        m['TNX_Val'] = df_macro['^TNX']
        m['TNX_Cambio'] = df_macro['^TNX'].diff()
    if '^VIX' in df_macro.columns:
        m['VIX_Val'] = df_macro['^VIX']
        m['VIX_Cambio'] = df_macro['^VIX'].diff()

    m.index = pd.to_datetime(m.index, utc=True).tz_localize(None).normalize()
    return m

def actualizar_data_lake():
    logger.info("="*60)
    logger.info(f"=== INICIANDO FUSIÓN MACROECONÓMICA Y CÁLCULO DE FEATURES ===")
    logger.info("="*60)

    archivos_parquet = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    if not archivos_parquet: return

    macro_df = obtener_datos_macro()
    tickers = sorted([os.path.basename(f).replace('.parquet', '') for f in archivos_parquet])
    
    data_nueva = yf.download(tickers, period='60d', progress=False)
    tickers_actualizados = 0
    cols_macro = ['SPY_Ret', 'TNX_Val', 'TNX_Cambio', 'VIX_Val', 'VIX_Cambio']

    for ticker in tickers:
        try:
            if len(tickers) > 1: df_reciente = data_nueva.xs(ticker, axis=1, level=1, drop_level=True).copy()
            else: df_reciente = data_nueva.copy()
                
            df_reciente.dropna(inplace=True)
            if df_reciente.empty: continue

            filepath = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
            df_historico = pd.read_parquet(filepath)
            
            if 'Date' not in df_historico.columns and 'index' in df_historico.columns: df_historico = df_historico.rename(columns={'index': 'Date'})
            elif 'Date' not in df_historico.columns:
                df_historico = df_historico.reset_index()
                if 'index' in df_historico.columns: df_historico = df_historico.rename(columns={'index': 'Date'})
            
            df_historico['Date'] = pd.to_datetime(df_historico['Date'], utc=True).dt.tz_localize(None)
            df_historico.set_index('Date', inplace=True)

            # --- LA CURA: Extraer SOLO los datos crudos antes de fusionar ---
            cols_base = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Evita errores si falta alguna columna base
            cols_historico = [c for c in cols_base if c in df_historico.columns]
            cols_reciente = [c for c in cols_base if c in df_reciente.columns]
            
            df_historico_base = df_historico[cols_historico].copy()
            df_reciente_base = df_reciente[cols_reciente].copy()

            df_reciente_base.index = pd.to_datetime(df_reciente_base.index, utc=True).tz_localize(None)
            df_fusionado = pd.concat([df_historico_base[df_historico_base.index < df_reciente_base.index[0]], df_reciente_base])
            df_fusionado = df_fusionado[~df_fusionado.index.duplicated(keep='last')].sort_index()

            # --- APLICAMOS TU CÓDIGO MAESTRO ---
            # Esto reconstruye todos los indicadores al 100% sobre la data fusionada sin dejar NaNs sueltos
            df_enriquecido = FeatureEngineer.apply_features(df_fusionado)
            
            # --- INYECCIÓN MACRO ---
            df_enriquecido.index = df_enriquecido.index.normalize()
            df_final = df_enriquecido.join(macro_df[cols_macro], how='left')
            df_final[cols_macro] = df_final[cols_macro].ffill().bfill()
            
            # Limpieza final pura
            df_final.dropna(inplace=True)

            df_final = df_final.reset_index().rename(columns={'index': 'Date'})
            df_final.to_parquet(filepath, index=False)
            tickers_actualizados += 1

        except Exception as e:
            logger.error(f"Error procesando {ticker}: {e}")

    logger.info("="*60)
    logger.info(f"✅ Proceso Finalizado. {tickers_actualizados}/{len(tickers)} archivos sincronizados al día de HOY.")
    logger.info("="*60)

if __name__ == "__main__":
    actualizar_data_lake()