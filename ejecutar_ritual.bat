@echo off
echo === INICIANDO PIPELINE DEL ORACULO V2 ===
:: Nos movemos a la carpeta del proyecto
cd /d "C:\Users\rolda\Oraculo V2"

echo 1/5 Actualizando datos (Ingesta)...
python src/ingesta.py

echo 2/5 Construyendo Features...
python src/features/build_features.py

echo 3/5 Sincronizando Macro...
python src/updater.py

echo 4/5 Re-entrenando IA...
python src/models/train_xgboost.py

echo 5/5 Enviando a Telegram...
python src/utils/telegram_bot.py

echo 6/6 Ejecutando operacion en Wall Street (Broker)...
python src/trading/ejecutor_quant.py

echo === TODO SINCRONIZADO ===
pause