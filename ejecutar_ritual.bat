@echo off
echo === INICIANDO PIPELINE DEL ORACULO V2 ===
:: Nos movemos a la carpeta del proyecto
cd /d "C:\Users\rolda\Oraculo V2"

echo 1/7 Actualizando datos (Ingesta)...
python src/ingesta.py

echo 2/7 Construyendo Features...
python src/features/build_features.py

echo 3/7 Sincronizando Macro...
python src/updater.py

echo 4/7 Re-entrenando IA...
python src/models/train_xgboost.py

echo 5/7 Ejecutando operacion en Wall Street (Broker)...
python src/trading/ejecutor_quant.py

echo 6/7 Enviando reporte final a Telegram...
python src/utils/telegram_bot.py

echo === TODO SINCRONIZADO ===
pause