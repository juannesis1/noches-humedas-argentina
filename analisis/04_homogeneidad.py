"""Homogeneidad por comparación con vecinas (en la línea de Willett et al. 2014).

Para cada estación candidata y cada variable (DPD y T, como HadISDH; también Td):
1. Serie mensual de anomalías nocturnas (meses oct-mar, 1980-2025).
2. Referencia = mediana de las anomalías de hasta 5 vecinas a <500 km con r > 0.5
   (correlación de las primeras diferencias, para no premiar tendencias compartidas).
3. Serie diferencia = candidata - referencia.
4. Test SNHT de un salto (Alexandersson 1986) con valor crítico por Monte Carlo (95 %),
   aplicado de forma recursiva (segmentación binaria, segmentos ≥24 meses).
Se reportan los quiebres y su magnitud. Las vecinas también pueden tener quiebres: por eso
se usa la mediana de varias y se informa la coherencia entre ellas.
"""
import glob
import os

import numpy as np
import pandas as pd

VARS = ["dpd", "t", "td"]
MESES = [10, 11, 12, 1, 2, 3]
RNG = np.random.default_rng(42)
_CRIT = {}


def mensual(sid):
    d = pd.read_parquet(f"{os.environ.get('DIR_BASE', 'data/noches')}/{sid}.parquet")
    d = d[(d.index.year >= 1979) & (d.index.year <= 2025)]
    m = d[VARS].resample("MS").mean()
    n = d[VARS[0]].resample("MS").count()
    m[n < 10] = np.nan
    m = m[m.index.month.isin(MESES)]
    m = m[(m.index >= "1979-10-01") & (m.index <= "2025-03-01")]
    clim = m.groupby(m.index.month).transform("mean")
    return m - clim


def snht(x):
    """Estadístico T máximo de SNHT y su posición."""
    z = (x - x.mean()) / x.std(ddof=1)
    n = len(z)
    cs = np.cumsum(z)
    k = np.arange(1, n)
    z1 = cs[:-1] / k
    z2 = (cs[-1] - cs[:-1]) / (n - k)
    t = k * z1 ** 2 + (n - k) * z2 ** 2
    t[:12] = 0
    t[-12:] = 0
    i = int(np.argmax(t))
    return t[i], i + 1


def ar1(n, phi):
    e = RNG.normal(size=n + 50)
    x = np.zeros_like(e)
    for i in range(1, len(e)):
        x[i] = phi * x[i - 1] + e[i]
    return x[50:]


def critico(n, phi):
    """Valor crítico 95 % de SNHT bajo ruido AR(1) con la autocorrelación de la serie
    (en el espíritu de Wang 2008: sin esto, la autocorrelación produce falsos quiebres)."""
    clave = (n, round(phi, 2))
    if clave not in _CRIT:
        sims = [snht(ar1(n, clave[1]))[0] for _ in range(1000)]
        _CRIT[clave] = np.percentile(sims, 95)
    return _CRIT[clave]


def phi_lag1(x):
    x = x - x.mean()
    return float(np.clip(np.sum(x[1:] * x[:-1]) / np.sum(x * x), 0, 0.9))


def segmentar(x, idx, out, prof=0):
    x = np.asarray(x)
    if len(x) < 48 or prof > 3:
        return
    t, k = snht(x)
    # autocorrelación estimada con el salto candidato removido
    res = np.concatenate([x[:k] - x[:k].mean(), x[k:] - x[k:].mean()])
    if t > critico(len(x), phi_lag1(res)):
        out.append((idx[k], x[k:].mean() - x[:k].mean(), t))
        segmentar(x[:k], idx[:k], out, prof + 1)
        segmentar(x[k:], idx[k:], out, prof + 1)


def distancia_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    return 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2)
                                        * np.sin((lo2 - lo1) / 2) ** 2))


def main():
    info = pd.read_fwf("data/hadisd/station_fullinfo.txt", header=None,
                       colspecs=[(0, 12), (13, 43), (43, 51), (51, 60), (60, 68)],
                       names=["sid", "nombre", "lat", "lon", "elev"]).set_index("sid")
    tend = pd.read_csv("analisis/03_tendencias_anomalias.csv")
    candidatas = tend[(tend.temporada == "ONDJFM") & tend.incluida].sid.tolist()
    todas = [os.path.basename(f)[:-8] for f in glob.glob("data/noches/*.parquet")]
    series = {s: mensual(s) for s in todas}
    filas = []
    for c in candidatas:
        pc = (info.loc[c, "lat"], info.loc[c, "lon"])
        for v in VARS:
            xc = series[c][v]
            vec = []
            for s in todas:
                if s == c:
                    continue
                dkm = distancia_km(pc, (info.loc[s, "lat"], info.loc[s, "lon"]))
                if dkm > 500:
                    continue
                xs = series[s][v].reindex(xc.index)
                par = pd.concat([xc.diff(), xs.diff()], axis=1).dropna()
                if len(par) < 120:
                    continue
                r = par.corr().iloc[0, 1]
                if r > 0.5:
                    vec.append((r, s, dkm))
            vec = sorted(vec, reverse=True)[:5]
            fila = {"sid": c, "nombre": info.loc[c, "nombre"], "var": v, "n_vecinas": len(vec),
                    "vecinas": ";".join(f"{info.loc[s, 'nombre'].strip()}({r:.2f},{d:.0f}km)"
                                        for r, s, d in vec),
                    "vecinas_sid": ";".join(s for _, s, _ in vec)}
            if len(vec) >= 2:
                ref = pd.concat([series[s][v].reindex(xc.index) for _, s, _ in vec], axis=1)
                ref = ref.median(axis=1, skipna=True).where(ref.notna().sum(axis=1) >= 2)
                dif = (xc - ref).dropna()
                quiebres = []
                segmentar(dif.values, dif.index, quiebres)
                quiebres.sort()
                fila["quiebres"] = "; ".join(f"{q[0]:%Y-%m} ({q[1]:+.2f} °C)" for q in quiebres)
                fila["n_quiebres"] = len(quiebres)
                # tendencia de la serie diferencia (°C/década): tendencia no compartida con vecinas
                x = (dif.index.year + dif.index.month / 12).values
                fila["tend_dif_dec"] = np.polyfit(x, dif.values, 1)[0] * 10
            filas.append(fila)
            print(c, v, fila.get("quiebres", "SIN VECINAS"), flush=True)
    pd.DataFrame(filas).to_csv("analisis/04_homogeneidad.csv", index=False)


if __name__ == "__main__":
    main()
