import numpy as np
import pandas as pd

def ejecutar_monte_carlo(df, dias_proyeccion=30, n_simulaciones=1000, vol_mult=1.0):
    if df.empty or len(df) < 20:
        return pd.DataFrame(), {}
        
    # FIX DEFINITIVO: Usar estrictamente 'Close'. Ignorar 'Adj Close' para evitar saltos.
    precios = df['Close'].ffill().dropna()
    
    if len(precios) < 20:
        return pd.DataFrame(), {}
        
    retornos_log = np.log(precios / precios.shift(1)).dropna()
    mu = retornos_log.mean() 
    sigma_base = retornos_log.std()
    
    if pd.isna(sigma_base) or sigma_base == 0:
        sigma_base = 0.001
        
    sigma_adj = sigma_base * vol_mult
    drift_adj = mu - (0.5 * (sigma_adj**2))
    
    # S0 ahora será EXACTAMENTE igual a la punta de tu gráfico
    S0 = precios.iloc[-1]
    
    rng = np.random.default_rng() 
    shocks = rng.normal(0, 1, (dias_proyeccion, n_simulaciones))
    
    incrementos = np.exp(drift_adj + sigma_adj * shocks)
    caminos = np.vstack([np.full(n_simulaciones, S0), incrementos])
    precios_simulados = np.cumprod(caminos, axis=0)
    
    df_sim = pd.DataFrame(precios_simulados)
    precios_finales = precios_simulados[-1, :]
    
    p10, p50, p90 = np.percentile(precios_finales, [10, 50, 90])
    spread_rel = (p90 - p10) / S0 if S0 != 0 else 0
    
    factor_tiempo = np.sqrt(dias_proyeccion / 7.0)
    score_confianza = max(0, min(100, 100 * (1 - (spread_rel / (0.6 * factor_tiempo)))))
    
    driver_drift = abs(drift_adj)
    driver_sigma = sigma_base
    driver_stress = sigma_adj - sigma_base
    
    total_impact = driver_drift + driver_sigma + driver_stress
    if total_impact == 0: total_impact = 1
    
    top_drivers = [
        {"name": "Tendencia Histórica", "impact": (driver_drift / total_impact) * 100},
        {"name": "Volatilidad Base", "impact": (driver_sigma / total_impact) * 100},
        {"name": "Factor de Estrés", "impact": (driver_stress / total_impact) * 100}
    ]
    top_drivers = sorted(top_drivers, key=lambda x: x['impact'], reverse=True)
    
    stats = {
        "current_price": S0,
        "mean_projection": np.mean(precios_finales),
        "p50": p50,
        "p10": p10,
        "p90": p90,
        "std_dev": np.std(precios_finales),
        "score_confianza": score_confianza,
        "driver_principal": top_drivers[0]['name'],
        "sensibilidad": sigma_adj / sigma_base,
        "dias": dias_proyeccion
    }
    
    return df_sim, stats