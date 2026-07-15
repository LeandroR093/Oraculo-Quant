@echo off
echo ====================================================
echo   🔮 INICIANDO PIPELINE DEL ORACULO V2 (SWING) 🔮
echo ====================================================
:: Nos movemos a la carpeta del proyecto
cd /d "C:\Users\rolda\Oraculo V2"

echo.
echo [1/7] Actualizando datos (Ingesta)...
python src/ingesta.py

echo.
echo [2/7] Construyendo Features e Indicadores...
python src/features/build_features.py

echo.
echo [3/7] Sincronizando datos Macro y VIX...
python src/updater.py

echo.
echo [4/7] Re-entrenando modelo IA (XGBoost)...
python src/models/train_xgboost.py

echo.
echo [5/7] Generando señales y calculando TP/SL (+2%% / -1.5%%)...
python src/trading/ejecutor_quant.py

echo.
echo [6/7] Difundiendo alertas al Canal de Telegram...
python src/utils/telegram_bot.py

echo.
echo [7/7] Subiendo datos actualizados a GitHub y Render...
:: Añadimos solo las carpetas de datos y código a Git (respeta tu .gitignore)
git add .
:: Creamos el commit de forma silenciosa con un mensaje automático
git commit -m "🤖 Auto-update diario Oráculo: Señales y datos"
:: Subimos a GitHub (esto dispara la actualización en Render)
git push origin main

echo.
echo ====================================================
echo   ✅ RITUAL COMPLETADO: SEÑALES Y WEB ACTUALIZADAS
echo ====================================================
pause