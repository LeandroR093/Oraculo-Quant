import os
import gc
import glob
import logging
import pandas as pd
import numpy as np

# Core library Quantitative de Technical Analysis
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volume import ChaikinMoneyFlowIndicator
from ta.volatility import BollingerBands

# ==============================================================================
# CONFIGURACIÓN DE LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_engineering.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# LÓGICA DE CÁLCULO VECTORIZADO DE FEATURES (V1 + V2 REFACTORED)
# ==============================================================================

class FeatureEngineer:
    """
    Clase para inyectar indicadores técnicos combinando la librería institucional `ta`
    con el procesamiento numérico y estadístico de Pandas optimizado.
    """
    
    @staticmethod
    def apply_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera matriz de features requeridas usando OHLCV, evadiendo ciclos FOR.
        """
        df = df.copy()

        # Validamos columnas requeridas para indicadores de Precio y Volumen
        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in req_cols):
            logger.warning("Faltan columnas mínimas requeridas. Calculando solo básicos permitidos.")
            return df
            
        close_col = df['Close']
        high_col = df['High']
        low_col = df['Low']
        vol_col = df['Volume']

        # ----------------------------------------------------------------------
        # 1. Ingeniería Vectorizada desde Pandas y Numpy Base
        # ----------------------------------------------------------------------
        
        # Estadísticas de 20 Días (Desviación y Z-Score)
        sma_20 = close_col.rolling(window=20).mean()
        df['Std_20'] = close_col.rolling(window=20).std()
        df['Z_Score_20'] = (close_col - sma_20) / df['Std_20']

        # Momentum de 10 días bursátiles
        df['Momentum_10'] = close_col.diff(10)

        # Contexto analítico de Volumen de transacciones
        df['Vol_SMA_20'] = vol_col.rolling(window=20).mean()
        # Uso de np.where genérico para evitar dividir por cero infinito en algunas paradas bursátiles extrañas
        df['Vol_Ratio'] = np.where(df['Vol_SMA_20'] == 0, 0, vol_col / df['Vol_SMA_20'])
        df['Vol_Change'] = vol_col.pct_change()

        # Resiliencia de precio (Pct from Highest High 50)
        high_50 = high_col.rolling(window=50).max()
        df['Pct_From_High_50'] = (close_col - high_50) / high_50

        # Métrica Base: Retornos Diarios
        df['Daily_Return'] = close_col.pct_change()
        
        # Retenemos EMA Predictiva Clásica (usada en gráficos UI y modelos a largo plazo)
        df['EMA_50'] = close_col.ewm(span=50, adjust=False).mean()
        df['EMA_200'] = close_col.ewm(span=200, adjust=False).mean()

        # ----------------------------------------------------------------------
        # 2. Indicadores Bursátiles Avanzados mediante `ta` library
        # ----------------------------------------------------------------------
        
        # Limpieza previa de Infinity generados por divisiones manuales atípicas
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fuerza Relativa (RSI 14)
        df['RSI_14'] = RSIIndicator(close=close_col, window=14).rsi()
        
        # MACD Engine Complete (Line, Signal y Diferencia Cuántica Histórica)
        macd = MACD(close=close_col, window_slow=26, window_fast=12, window_sign=9)
        df['MACD_Line'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Tendencia Direccional (ADX 14)
        df['ADX'] = ADXIndicator(high=high_col, low=low_col, close=close_col, window=14).adx()
        
        # Chaikin Money Flow (CMF 20) inyectando Peso Histórico de Volumen vs Precio
        df['CMF'] = ChaikinMoneyFlowIndicator(high=high_col, low=low_col, close=close_col, volume=vol_col, window=20).chaikin_money_flow()
        
        # Volatilidad a Corredores 2xDesviaciones (Bollinger Bands High y Low)
        bb = BollingerBands(close=close_col, window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()

        # ----------------------------------------------------------------------
        # 3. Profiling Dimensional Definitivo (Limpiezas y Purgas Null)
        # ----------------------------------------------------------------------
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Se desecharán los datos crudos iniciales por no alcanzar madurez (ej: 200 días para EMA)
        df.dropna(inplace=True)

        return df

# ==============================================================================
# MOTOR ITERATIVO SOBRE DATA LAKE LOCAL (Fase RAW a PROCESSED)
# ==============================================================================

def main():
    logger.info("="*60)
    logger.info("=== Iniciando Iteración Fase 2: Construcción Features V1/V2 ===")
    logger.info("="*60)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')

    os.makedirs(processed_dir, exist_ok=True)

    raw_files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    total_files = len(raw_files)
    
    if total_files == 0:
        logger.error("No se hallaron archivos `.parquet`. Revisa la Fase 1.")
        return

    logger.info(f"Injectando Machine Learning Library a {total_files} activos. El proceso iniciará...")

    procesados_con_exito = 0

    for filepath in raw_files:
        filename = os.path.basename(filepath)
        ticker = filename.replace('.parquet', '')
        
        try:
            df_crudo = pd.read_parquet(filepath)
            
            if df_crudo.empty:
                continue
                
            # Procesamiento Vectorial Pura
            df_enriched = FeatureEngineer.apply_features(df_crudo)
            
            if df_enriched.empty:
                logger.warning(f"[{ticker}] DESCARTADO: Escasez de historial (<200 ds) para cálculo de modelos formales.")
                continue

            # Serializado Snappy
            dest_filepath = os.path.join(processed_dir, filename)
            df_enriched.to_parquet(dest_filepath, engine='pyarrow', compression='snappy')
            
            procesados_con_exito += 1

        except Exception as e:
            logger.error(f"[{ticker}] Falla Crítica en Ingeniería: {e}")

        finally:
            if 'df_crudo' in locals(): del df_crudo
            if 'df_enriched' in locals(): del df_enriched
            gc.collect()

    logger.info("="*60)
    logger.info(f"=== Operación Fase 2 Exitosa. Total Exportados: {procesados_con_exito}/{total_files} ===")
    logger.info("="*60)

if __name__ == "__main__":
    main()
