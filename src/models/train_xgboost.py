import os
import gc
import glob
import logging
import joblib

import pandas as pd
import numpy as np
import ta
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("=== Iniciando Fase 3: Modelo Quant Institucional (High-Alpha) ===")
    logger.info("="*60)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    outputs_dir = os.path.join(base_dir, 'data', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    parquet_files = glob.glob(os.path.join(processed_dir, "*.parquet"))
    if not parquet_files: return
        
    logger.info(f"Cargando e inyectando memoria Quant a {len(parquet_files)} activos...")
    
    df_list = []
    for f in parquet_files:
        ticker = os.path.basename(f).replace('.parquet', '')
        try:
            temp_df = pd.read_parquet(f)
            if not temp_df.empty:
                if 'Date' not in temp_df.columns:
                    temp_df = temp_df.reset_index()
                    if 'index' in temp_df.columns: temp_df.rename(columns={'index': 'Date'}, inplace=True)
                
                if 'Close' in temp_df.columns:
                    temp_df['Close'] = temp_df['Close'].ffill()
                    
                    # --- RESTAURACIÓN DEL MOTOR QUANT Y MEMORIA ---
                    temp_df['Retorno_Diario'] = temp_df['Close'].pct_change()
                    temp_df['RSI'] = ta.momentum.rsi(temp_df['Close'], window=14)
                    
                    bb_upper = ta.volatility.bollinger_hband(temp_df['Close'], window=20, window_dev=2)
                    bb_lower = ta.volatility.bollinger_lband(temp_df['Close'], window=20, window_dev=2)
                    temp_df['BB_Posicion'] = (temp_df['Close'] - bb_lower) / (bb_upper - bb_lower)
                    
                    temp_df['MACD_Hist'] = ta.trend.macd_diff(temp_df['Close'], window_fast=12, window_slow=26, window_sign=9)
                    
                    sma_20 = ta.trend.sma_indicator(temp_df['Close'], window=20)
                    sma_50 = ta.trend.sma_indicator(temp_df['Close'], window=50)
                    sma_200 = ta.trend.sma_indicator(temp_df['Close'], window=200) # NUEVO: Filtro Macro
                    
                    temp_df['Dist_SMA20'] = (temp_df['Close'] / sma_20) - 1
                    temp_df['Dist_SMA50'] = (temp_df['Close'] / sma_50) - 1
                    temp_df['Dist_SMA200'] = (temp_df['Close'] / sma_200) - 1 # NUEVO: Mean Reversion
                    
                    # NUEVO: Sensor de Volatilidad a corto plazo
                    temp_df['Volatilidad_5d'] = temp_df['Retorno_Diario'].rolling(window=5).std()
                    
                    # NUEVO: Sensor de Pánico (Aceleración del VIX)
                    if 'VIX_Val' in temp_df.columns:
                        temp_df['VIX_Tendencia'] = temp_df['VIX_Val'] / temp_df['VIX_Val'].rolling(window=20).mean() - 1

                    temp_df['Retorno_Ayer'] = temp_df['Retorno_Diario'].shift(1)
                    temp_df['Retorno_Anteayer'] = temp_df['Retorno_Diario'].shift(2)
                    temp_df['RSI_Ayer'] = temp_df['RSI'].shift(1)
                    temp_df['MACD_Hist_Ayer'] = temp_df['MACD_Hist'].shift(1)
                    
                temp_df['Ticker'] = ticker
                df_list.append(temp_df)
        except Exception as e: pass
            
    df = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()
    
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

    df['Ticker'] = df['Ticker'].astype('category')

    # ==========================================================================
    # LA MAGIA DEL "SMART TARGET" (Solo operamos si vale la pena)
    # ==========================================================================
    df.sort_values(by=['Ticker', 'Date'], inplace=True)
    df['Next_Close'] = df.groupby('Ticker')['Close'].shift(-1)
    
    # Calculamos cuánto va a subir exactamente mañana
    df['Retorno_Futuro'] = (df['Next_Close'] / df['Close']) - 1
    
    # UMBRAL EXIGENTE: Solo es un "1" si la acción sube más de un +0.25% en un día.
    # Si sube poco, se queda plana o baja, es un "0".
    df['Target'] = (df['Retorno_Futuro'] > 0.0025).astype(int)
    
    df.dropna(subset=['Next_Close'], inplace=True)
    df.drop(columns=['Next_Close', 'Retorno_Futuro'], inplace=True)
    
    # Amputamos dólares crudos para evitar sobreajuste
    columnas_crudas = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Capital Gains', 'Dividends', 'Stock Splits', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'BB_Mid', 'SMA_20', 'SMA_50']
    df.drop(columns=[c for c in columnas_crudas if c in df.columns], inplace=True, errors='ignore')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    logger.info(f"Registros limpios (High-Alpha): {len(df)}")
    
    cols_to_drop = ['Date', 'Target'] 
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    # TIME-SERIES SPLIT
    split_date = pd.to_datetime('2024-01-01').tz_localize(None)
    df_train = df[df['Date'] < split_date].copy()
    df_test  = df[df['Date'] >= split_date].copy()
    
    X_train, y_train = df_train[feature_cols], df_train['Target']
    X_test, y_test   = df_test[feature_cols], df_test['Target']
    
    del df, df_train, df_test
    gc.collect()
    
    # XGBOOST CATEGÓRICO AFINADO
    logger.info("Entrenando Motor de Inferencia Quant...")
    clase_0 = (y_train == 0).sum()
    clase_1 = (y_train == 1).sum()
    
    # Como ahora la clase 1 es más rara (movimientos fuertes), el peso subirá naturalmente
    scale_pos = clase_0 / clase_1 if clase_1 > 0 else 1.0
    logger.info(f"Distribución -> Días Planos/Caídas: {clase_0} | Días de Alta Subida (+0.25%): {clase_1}")
    
    model = XGBClassifier(
        n_estimators=400,          # Le damos un poco más de tiempo para pensar
        max_depth=5,               # Reducimos profundidad para evitar memorizar ruido (Overfitting)
        learning_rate=0.03,        # Pasos más pequeños y seguros
        subsample=0.8,             
        colsample_bytree=0.8,      
        scale_pos_weight=scale_pos,
        enable_categorical=True,    
        tree_method='hist',         
        random_state=42,           
        n_jobs=-1                  
    )
    model.fit(X_train, y_train)
    
    logger.info("Evaluando Blind-Test 2024+ ...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] 
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*55)
    print("====== REPORTE CLASIFICATORIO (BLIND-TEST 2024+) ======")
    print("="*55)
    print(classification_report(y_test, y_pred))
    print(f">> PODER PREDICTIVO EN CURVA ROC-AUC: {roc_auc:.4f} <<")
    print("="*55 + "\n")
    
    model_path = os.path.join(outputs_dir, 'oraculo_xgboost.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Modelo exitosamente guardado en: {model_path}")
    
if __name__ == "__main__":
    main()