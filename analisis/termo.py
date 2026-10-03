"""Termodinámica húmeda para el paper.

Tw (temperatura de bulbo húmedo pseudoadiabática, Davies-Jones 2008): es la temperatura
de una parcela saturada, a la presión de la estación, cuya temperatura potencial
equivalente (Bolton 1980, Ec. 39) es igual a la del aire observado. En lugar de la
aproximación inicial + una iteración de Davies-Jones, resolvemos esa ecuación con
Newton hasta convergencia (|ΔTw| < 1e-4 K), lo que da el mismo resultado sin depender
de la forma de las aproximaciones.
"""
import numpy as np

KAPPA = 0.2854      # R_d / c_pd (Bolton 1980)
EPS = 0.622
P0 = 1000.0         # hPa
C = 273.15


def es_bolton(t_c):
    """Presión de vapor de saturación sobre agua (hPa), Bolton (1980) Ec. 10. t en °C."""
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def presion_estandar(z_m):
    """Presión (hPa) a la altura z con atmósfera estándar (como Raymond et al. 2020)."""
    return 1013.25 * (1 - 2.25577e-5 * z_m) ** 5.25588


def theta_e(t_c, td_c, p):
    """Temperatura potencial equivalente (K): θDL de Bolton (1980) y su exponencial
    asociada (Davies-Jones 2008), con TL de Bolton Ec. 15."""
    T = t_c + C
    Td = td_c + C
    e = es_bolton(td_c)
    r = 1000 * EPS * e / (p - e)                     # g/kg
    TL = 1 / (1 / (Td - 56) + np.log(T / Td) / 800) + 56
    th_dl = T * (P0 / (p - e)) ** KAPPA * (T / TL) ** (0.28e-3 * r)
    # Constantes de la forma con θDL (Bolton 1980; Davies-Jones 2008; Buzan et al. 2015 Ec. A4)
    return th_dl * np.exp((3.036 / TL - 0.001788) * r * (1 + 0.448e-3 * r))


def theta_e_sat(tw_c, p):
    """θE de una parcela saturada a tw (°C) y p (hPa)."""
    return theta_e(tw_c, tw_c, p)


def tw_davies_jones(t_c, td_c, p, tol=1e-4, maxit=50):
    """Bulbo húmedo pseudoadiabático (°C) resolviendo θE_sat(Tw, p) = θE(T, Td, p)."""
    t_c, td_c, p = np.broadcast_arrays(np.asarray(t_c, float), np.asarray(td_c, float),
                                       np.asarray(p, float))
    objetivo = theta_e(t_c, td_c, p)
    tw = t_c - (t_c - td_c) / 3.0                    # arranque: regla del tercio
    for _ in range(maxit):
        f = theta_e_sat(tw, p) - objetivo
        h = 1e-3
        df = (theta_e_sat(tw + h, p) - theta_e_sat(tw - h, p)) / (2 * h)
        paso = f / df
        tw = tw - paso
        if np.nanmax(np.abs(paso)) < tol:
            break
    return np.where(np.isfinite(t_c) & np.isfinite(td_c), tw, np.nan)


def tw_stull(t_c, rh):
    """Stull (2011), válido a 1013 hPa. Solo para sensibilidad."""
    return (t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t_c + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)


def rh_desde_td(t_c, td_c):
    return 100 * es_bolton(td_c) / es_bolton(t_c)


def q_desde_td(td_c, p):
    """Humedad específica (g/kg)."""
    e = es_bolton(td_c)
    return 1000 * EPS * e / (p - (1 - EPS) * e)
