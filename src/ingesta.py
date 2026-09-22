import os
import gc
import time
import random
import logging
from datetime import timedelta
from abc import ABC, abstractmethod
from typing import List, Tuple

import requests
from io import StringIO

import pandas as pd
import yfinance as yf

# ==============================================================================
# CONFIGURACIÓN DE LOGGING
# ==============================================================================
# Logs visibles en terminal y guardados en 'ingestion.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ingestion.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# PATRÓN STRATEGY PARA ALMACENAMIENTO (PREPARADO PARA CLOUD - FASE 2)
# ==============================================================================

class StorageBackend(ABC):
    @abstractmethod
    def exists(self, identifier: str) -> bool:
        pass
    
    @abstractmethod
    def save(self, df: pd.DataFrame, identifier: str) -> None:
        pass

    @abstractmethod
    def load(self, identifier: str) -> pd.DataFrame:
        pass

class LocalParquetStorage(StorageBackend):
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        
    def _get_filepath(self, identifier: str) -> str:
        # Formatear el nombre del archivo para que sea un path seguro en cualquier OS
        safe_id = identifier.replace('^', '').replace('=', '')
        return os.path.join(self.base_path, f"{safe_id}.parquet")

    def exists(self, identifier: str) -> bool:
        return os.path.exists(self._get_filepath(identifier))

    def save(self, df: pd.DataFrame, identifier: str) -> None:
        filepath = self._get_filepath(identifier)
        # Guardamos usando pyarrow y compresión snappy (requerimiento institucional)
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        logger.debug(f"Guardado exitosamente en crudo: {filepath}")

    def load(self, identifier: str) -> pd.DataFrame:
        filepath = self._get_filepath(identifier)
        # Leemos el archivo parquet existente
        return pd.read_parquet(filepath, engine='pyarrow')

# ==============================================================================
# LÓGICA DE EXTRACCIÓN DE TICKERS S&P 500 Y GICS SECTOR
# ==============================================================================

def get_sp500_tickers_and_sectors() -> List[Tuple[str, str]]:
    """
    Extrae la lista de constituyentes del S&P 500 y su sector GICS desde la URL de Wikipedia.
    """
    logger.info("Extrayendo lista de constituyentes del S&P 500 desde Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Lanza excepción si el código no es 200
        
        # Envolvemos text en StringIO para compatibilidad con Pandas 2+
        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]
        
        tickers_data = []
        for _, row in sp500_table.iterrows():
            ticker = str(row['Symbol'])
            # Reemplazar puntos por guiones, convención de Yahoo Finance (ej. BRK.B -> BRK-B)
            ticker = ticker.replace('.', '-')
            sector = str(row['GICS Sector'])
            tickers_data.append((ticker, sector))
        
        logger.info(f"Se obtuvieron {len(tickers_data)} activos del S&P 500 exitosamente.")
        return tickers_data
    except Exception as e:
        logger.error(f"Error al obtener la tabla del S&P 500 desde Wikipedia: {e}")
        return []

# ==============================================================================
# WORKFLOW PRINCIPAL DE INGESTA
# ==============================================================================

class DataIngestionPipeline:
    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def fetch_and_save(self, ticker: str, sector: str = "Unknown"):
        """
        Descarga el historial de un ticker. Si ya existe, hace una sincronización incremental.
        """
        df_existente = None
        fecha_inicio = None

        # 1. VERIFICAR Y LEER LA ÚLTIMA FECHA
        if self.storage.exists(ticker):
            try:
                df_existente = self.storage.load(ticker)
                
                # Asumiendo que el índice de yfinance es la fecha (DatetimeIndex)
                ultima_fecha = pd.to_datetime(df_existente.index.max())
                
                # Sumamos 1 día para no duplicar el último día que ya tenemos
                fecha_inicio = (ultima_fecha + timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"[{ticker}] Archivo encontrado. Actualizando desde {fecha_inicio}...")
                
            except Exception as e:
                logger.warning(f"[{ticker}] No se pudo leer el archivo existente ({e}). Se forzará descarga completa.")
        else:
            logger.info(f"[{ticker}] Iniciando descarga completa por primera vez... | Sector: {sector}")

        try:
            # Rate Limiter Aleatorio (1 a 3 segundos) para evitar ban de IP (429 Too Many Requests)
            sleep_time = random.uniform(1.0, 3.0)
            time.sleep(sleep_time)

            ticker_obj = yf.Ticker(ticker)

            # 2. DESCARGAR (INCREMENTAL O COMPLETA)
            if fecha_inicio:
                # Descarga incremental
                df_nuevo = ticker_obj.history(start=fecha_inicio, auto_adjust=False)
                
                if df_nuevo.empty:
                    logger.info(f"[{ticker}] SALTADO - Ya está completamente al día.")
                    return
                
                # Asegurar que el nuevo DataFrame tenga la columna del sector antes de unir
                df_nuevo['GICS_Sector'] = sector
                
                # Unir datos viejos con nuevos
                df = pd.concat([df_existente, df_nuevo])
                # Eliminar duplicados por índice (fechas) por seguridad
                df = df[~df.index.duplicated(keep='last')]
                logger.info(f"[{ticker}] Sincronizado: Se añadieron {len(df_nuevo)} filas nuevas. Total: {len(df)}.")
                
            else:
                # Descarga completa original
                df = ticker_obj.history(period="max", auto_adjust=False)
                
                if df.empty:
                    logger.warning(f"[{ticker}] ADVERTENCIA - DataFrame vacío devuelto. El activo puede estar deslistado o ser inválido.")
                    return
                
                df['GICS_Sector'] = sector
                logger.info(f"[{ticker}] ÉXITO - {len(df)} registros procesados y guardados (Historial completo).")

            # Guardar el DataFrame final
            self.storage.save(df, ticker)

        except Exception as e:
            logger.error(f"[{ticker}] ERROR DURANTE DESCARGA - {e}")

        finally:
            # Optimizacion extrema de memoria: Limpiamos por cada iteración.
            if 'df' in locals():
                del df
            if 'df_existente' in locals():
                del df_existente
            if 'df_nuevo' in locals():
                del df_nuevo
            gc.collect()

def main():
    # 1. Configurar Backend de Storage (Fase 1: Local)
    # Apuntando a data/raw relativo a la ejecución para ser agnóstico del folder path.
    base_raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    storage = LocalParquetStorage(base_raw_dir)
    pipeline = DataIngestionPipeline(storage)

    # 2. Definir el Universo Institucional de Activos
    sp500_assets = get_sp500_tickers_and_sectors()
    
    # Activos Clave Adicionales requeridos
    additional_assets = [
        ("SPY", "ETF"),
        ("BTC-USD", "Crypto"),
        ("^VIX", "Index")
    ]

    # Unión de tuplas: (Ticker, Sector)
    universe = additional_assets + sp500_assets

    # FILTRO DE SEGURIDAD / LISTA NEGRA
    # ==============================================================================
    BLACKLIST = {"SATS", "CTRA"} 
    universe = [item for item in universe if item[0] not in BLACKLIST]

    # 3. Ejecutar Pipeline
    logger.info("="*60)
    logger.info("=== Iniciando Pipeline de Ingesta Oráculo V2 ===")
    logger.info("="*60)
    logger.info(f"Universo total a procesar y sincronizar: {len(universe)} activos.")
    
    for ticker, sector in universe:
        pipeline.fetch_and_save(ticker, sector)
        
    logger.info("="*60)
    logger.info("=== Trabajo Completado. Memoria Liberada. ===")
    logger.info("="*60)

if __name__ == "__main__":
    main()