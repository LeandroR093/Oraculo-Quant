import os
import pandas as pd
import boto3
from ib_insync import *
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

# --- CARGAR VARIABLES DE ENTORNO ---
load_dotenv()

# --- CONFIGURACIÓN AWS ---
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
S3_BUCKET_NAME = "oraculo-quant-storage-lroldan" # Tu bucket exacto

# --- RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
OUTPUT_CARTERA = os.path.join(ROOT_DIR, 'data', 'outputs', 'cartera_actual.csv')

def upload_to_s3(local_file_path: str, s3_key: str):
    """Sube el archivo al bucket de S3."""
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
    try:
        s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_key)
        print(f"☁️ Subido con éxito a S3: {s3_key}")
    except Exception as e:
        print(f"❌ Error al subir a S3: {e}")

def extraer_y_subir_cartera():
    print("📊 Extrayendo estado de la cartera de Interactive Brokers...")
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=4) # Usamos clientId=4 para no chocar con el ejecutor
        
        # 1. Obtener el saldo total de la cuenta (Net Liquidation Value)
        valores_cuenta = ib.accountSummary()
        saldo_total = 0.0
        efectivo_disponible = 0.0
        
        for val in valores_cuenta:
            if val.tag == 'NetLiquidation':
                saldo_total = float(val.value)
            elif val.tag == 'AvailableFunds':
                efectivo_disponible = float(val.value)

        # 2. Obtener las posiciones abiertas (Tus acciones)
        posiciones = ib.portfolio()
        lista_posiciones = []
        
        if not posiciones:
            print("🛡️ No hay posiciones abiertas. Todo está en efectivo.")
            lista_posiciones.append({
                'Ticker': 'CASH', 'Cantidad': 0, 'Precio_Compra': 0.0, 
                'Precio_Actual': 0.0, 'Ganancia_Perdida': 0.0, 
                'Valor_Total': efectivo_disponible
            })
        else:
            for pos in posiciones:
                lista_posiciones.append({
                    'Ticker': pos.contract.symbol,
                    'Cantidad': pos.position,
                    'Precio_Compra': round(pos.averageCost, 2),
                    'Precio_Actual': round(pos.marketPrice, 2),
                    'Ganancia_Perdida': round(pos.unrealizedPNL, 2),
                    'Valor_Total': round(pos.marketValue, 2)
                })
                
        # 3. Guardar en CSV Local
        df_cartera = pd.DataFrame(lista_posiciones)
        df_cartera['Saldo_Total_Cuenta'] = saldo_total # Añadimos el saldo global a todas las filas
        df_cartera.to_csv(OUTPUT_CARTERA, index=False)
        print(f"💾 Cartera guardada localmente en {OUTPUT_CARTERA}")
        
        # 4. Subir a la carpeta powerbi_data en Amazon S3
        s3_key = "powerbi_data/cartera_actual.csv"
        upload_to_s3(OUTPUT_CARTERA, s3_key)

    except Exception as e:
        print(f"❌ Error conectando a IBKR: {e}")
    finally:
        ib.disconnect()

if __name__ == "__main__":
    extraer_y_subir_cartera()