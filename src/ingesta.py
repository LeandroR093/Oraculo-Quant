import os
import gc
import time
import random
import logging
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
        Descarga todo el historial disponible para un ticker dado y lo guarda en el storage.
        """
        if self.storage.exists(ticker):
            logger.info(f"[{ticker}] SALTADO - Ya existe en la base de datos.")
            return

        logger.info(f"[{ticker}] Iniciando descarga... | Sector: {sector}")
        try:
            # Rate Limiter Aleatorio (1 a 3 segundos) para evitar ban de IP (429 Too Many Requests)
            sleep_time = random.uniform(1.0, 3.0)
            time.sleep(sleep_time)

            ticker_obj = yf.Ticker(ticker)
            # Extracción del Dataset completo al máximo historial.
            # Mantenemos auto_adjust=False para retener tanto "Close" crudo como "Adj Close" si existe.
            df = ticker_obj.history(period="max", auto_adjust=False)

            if df.empty:
                logger.warning(f"[{ticker}] ADVERTENCIA - DataFrame vacío devuelto. El activo puede estar deslistado o ser inválido.")
                return

            # Agregar el Sector GICS como feature base general (útil para Machine Learning)
            df['GICS_Sector'] = sector

            # Guardar el DataFrame
            self.storage.save(df, ticker)
            logger.info(f"[{ticker}] ÉXITO - {len(df)} registros procesados y guardados.")

        except Exception as e:
            logger.error(f"[{ticker}] ERROR DURANTE DESCARGA - {e}")

        finally:
            # Optimizacion extrema de memoria: Limpiamos por cada iteración.
            if 'df' in locals():
                del df
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
