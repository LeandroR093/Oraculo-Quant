# 📈 Oráculo Quant V2 - Automated ML Trading System

Un sistema de trading algorítmico (Quant) de extremo a extremo (End-to-End) diseñado para predecir "saltos alcistas" a corto plazo en el mercado de valores estadounidense (S&P 500) mediante Machine Learning, enviando alertas en tiempo real y ejecutando operaciones 100% automatizadas en Interactive Brokers.

## 🚀 Características Principales

* **Ingesta de Datos Autónoma:** Descarga y actualiza diariamente datos de mercado y variables macroeconómicas (VIX, SPY, TNX) a través de `yfinance`.
* **Feature Engineering Avanzado:** Cálculo automatizado de indicadores técnicos clave (RSI, Bandas de Bollinger, MACD, Medias Móviles) usando la librería `ta`.
* **Machine Learning (XGBoost):** Entrenamiento diario de un modelo de clasificación para identificar probabilidades de saltos alcistas (>0.25%) al cierre del mercado.
* **Notificaciones Push (Telegram):** Bot integrado que envía un reporte diario al móvil con el estado del VIX y el Top de acciones con mayor probabilidad de éxito.
* **Ejecución Algorítmica (IBKR):** Conexión vía API con Trader Workstation (TWS) para colocar automáticamente *Bracket Orders* (Compra, Take Profit al **+2%**, y Stop Loss al **-1.5%**) garantizando la diversificación del capital y la gestión del riesgo.
* **Orquestación Continua:** Pipeline completamente automatizado mediante un archivo Batch (`.bat`) y el Programador de Tareas de Windows.

## 🏗️ Arquitectura del Pipeline (El "Ritual Quant")

El sistema se ejecuta diariamente de forma autónoma antes del cierre del mercado siguiendo estos 6 pasos:

1.  **Ingesta (`ingesta.py`):** Actualización de la base de datos local con la última sesión de Wall Street.
2.  **Features (`build_features.py`):** Procesamiento de indicadores técnicos.
3.  **Macro (`updater.py`):** Sincronización del entorno macroeconómico (ej. VIX).
4.  **IA (`train_xgboost.py`):** Re-entrenamiento del modelo con los datos más frescos.
5.  **Alertas (`telegram_bot.py`):** Envío del reporte analítico por mensajería.
6.  **Trading (`ejecutor_quant.py`):** Selección del Top 2 de activos y envío de órdenes a Interactive Brokers.

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **Librerías de Datos:** Pandas, NumPy, yfinance, PyArrow (Parquet)
* **Machine Learning:** Scikit-Learn, XGBoost
* **Análisis Técnico:** TA-Lib (`ta`)
* **Integración Broker:** `ib_insync` (API de Interactive Brokers)
* **Automatización:** Windows Task Scheduler, Archivos Batch (.bat)

## 📁 Estructura del Proyecto

```text
Oraculo_V2/
│
├── data/                  # Almacenamiento local de datos (Raw, Processed, Outputs)
├── src/
│   ├── features/          # Scripts para cálculo de indicadores
│   ├── models/            # Entrenamiento y evaluación de ML (XGBoost)
│   ├── trading/           # Ejecutor algorítmico y conexión a IBKR (ib_insync)
│   ├── utils/             # Alertas de Telegram y utilidades
│   ├── ingesta.py         # Descarga de yfinance
│   └── updater.py         # Sincronización macro
│
├── ejecutar_ritual.bat    # Script maestro de automatización
└── README.md


## ⚠️ Aviso Legal (Disclaimer)
Este proyecto es estrictamente con fines educativos y de investigación. El código proporcionado no constituye asesoramiento financiero. El trading algorítmico conlleva un alto riesgo de pérdida de capital. Se recomienda encarecidamente utilizar este sistema única y exclusivamente en entornos de simulación (Paper Trading) y no con dinero real. El autor no se hace responsable de ninguna pérdida financiera incurrida por el uso de este software.