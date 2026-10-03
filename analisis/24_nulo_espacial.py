"""Significancia de la relación expansión agrícola – secado con un nulo espacial.

Las 20 estaciones no son independientes: la expansión y la tendencia de Td tienen estructura
espacial de gran escala. Dos enfoques (ver paper/revision_estricta.md, M1):
1. Tamaño muestral efectivo (Clifford et al. 1989; Dutilleul 1993), sobre rangos y sobre
   residuos de lat/lon.
2. Campos sustitutos (en el espíritu de Viladomat et al. 2014): campos gaussianos con la misma
   covarianza espacial (exponencial, ajustada al variograma de los rangos de la expansión)
   en las ubicaciones de las estaciones; la tendencia de Td se deja fija. p = fracción de
   campos simulados con |ρ| ≥ |ρ observado|. Sensibilidad al alcance del variograma (×0.5, ×2).
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

rng = np.random.default_rng(1)


def dist(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    c = np.sin(la[:, None]) * np.sin(la) + np.cos(la[:, None]) * np.cos(la) * np.cos(lo[:, None] - lo)
    return 6371 * np.arccos(np.clip(c, -1, 1))


def n_eff(x, y, D, paso=200):
    n = len(x)
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    s = 0.0
    for a in np.arange(0, D.max() + paso, paso):
        m = (D > a) & (D <= a + paso)
        if m.sum():
            i, j = np.where(m)
            s += m.sum() * np.mean(x[i] * x[j]) * np.mean(y[i] * y[j])
    return min(n, 1 / (1 / n + s / n ** 2))


def alcance(z, D):
    """Ajuste de un variograma exponencial γ(h) = c·(1 − e^(−h/a)) a la semivarianza empírica."""
    i, j = np.triu_indices(len(z), 1)
    h, g = D[i, j], 0.5 * (z[i] - z[j]) ** 2
    bins = np.arange(0, h.max() + 150, 150)
    hc = [h[(h >= a) & (h < b)].mean() for a, b in zip(bins[:-1], bins[1:]) if ((h >= a) & (h < b)).sum() >= 5]
    gc = [g[(h >= a) & (h < b)].mean() for a, b in zip(bins[:-1], bins[1:]) if ((h >= a) & (h < b)).sum() >= 5]
    f = lambda hh, c, a: c * (1 - np.exp(-hh / a))
    (c, a), _ = optimize.curve_fit(f, hc, gc, p0=[np.var(z), 300], bounds=([0, 20], [np.inf, 5000]))
    return a


def main():
    r = pd.read_csv("analisis/18_urbano.csv").dropna(subset=["delta_soja_pp"])
    D = dist(r.lat.values, r.lon.values)
    x = stats.rankdata(r.delta_soja_pp).astype(float)
    print(f"n = {len(r)} estaciones")
    for col, nombre in (("td_ond", "oct-dic"), ("td_sen", "oct-mar")):
        y = r[col].values
        rho = stats.spearmanr(x, y)[0]
        ne = n_eff(x, stats.rankdata(y), D)
        t = rho * np.sqrt((ne - 2) / (1 - rho ** 2))
        p_ne = 2 * stats.t.sf(abs(t), ne - 2)
        a = alcance(x, D)
        linea = f"{nombre}: ρ {rho:+.2f} | n_eff {ne:.1f}, p {p_ne:.3f} | alcance variograma {a:.0f} km"
        for f in (0.5, 1, 2):
            C = np.exp(-D / (a * f)) + 1e-9 * np.eye(len(x))
            L = np.linalg.cholesky(C)
            sims = L @ rng.standard_normal((len(x), 5000))
            rs = np.array([stats.spearmanr(sims[:, k], y)[0] for k in range(sims.shape[1])])
            linea += f" | sustitutos (alcance×{f}): p {np.mean(np.abs(rs) >= abs(rho)):.4f}"
        print(linea)


if __name__ == "__main__":
    main()
