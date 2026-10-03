"""Fecha de cambio de sensor de humedad estimada desde los datos (sustituto de metadatos).

Revisión estricta, M2: los psicrómetros reportan saturación (T = Td) con niebla o rocío; los sensores
capacitivos casi nunca. Sin metadatos del SMN, se estima por estación el mes en que la fracción
de observaciones nocturnas (06 y 09 UTC, todo el año para tener más datos) con depresión del punto
de rocío ≤ 0.2 °C cae de forma abrupta: punto de cambio único por máxima diferencia de medias
(estadístico SNHT) sobre la serie mensual, con valor crítico por permutación (999 réplicas por
bloques de 12 meses).
Luego, para cada estación con cambio detectado, tendencia de Td nocturna oct-dic (homogeneizada)
ANTES y DESPUÉS de esa fecha: si el secado agrícola fuera un artefacto del cambio de sensor,
aparecería como un escalón y no como tendencia dentro de cada segmento.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

rng = np.random.default_rng(11)


def snht_max(x):
    z = (x - x.mean()) / x.std()
    n = len(z)
    t = np.array([k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2 for k in range(12, n - 12)])
    return t.max(), int(t.argmax()) + 12


def fraccion_saturada(sid):
    nc = glob.glob(f"data/hadisd/nc/*_{sid}_humidity.nc")[0]
    d = xr.open_dataset(nc)
    df = pd.DataFrame({"t": d.temperatures.values, "td": d.dewpoints.values},
                      index=pd.DatetimeIndex(d.time.values))
    d.close()
    df = df[(df.index.minute == 0) & df.index.hour.isin([6, 9]) & (df.index.year >= 1979)].dropna()
    f = (df.t - df.td <= 0.2).resample("MS").mean()
    n = df.t.resample("MS").count()
    return f[n >= 20]


def main():
    tend = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = tend[(tend.temporada == "ONDJFM") & tend.incluida & ~tend.nombre.str.contains("CERES")]
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    h = __import__("importlib").import_module("19_huella_temporal")
    h.DIR = "data/noches_ajustadas_pares_laxo"
    filas = []
    for _, e in est.iterrows():
        f = fraccion_saturada(e.sid)
        anom = f - f.groupby(f.index.month).transform("mean")
        x = anom.values
        tmax, k = snht_max(x)
        bloques = [x[i:i + 12] for i in range(0, len(x), 12)]
        nulos = []
        for _ in range(999):
            perm = np.concatenate([bloques[j] for j in rng.permutation(len(bloques))])
            nulos.append(snht_max(perm)[0])
        p = np.mean(np.array(nulos) >= tmax)
        fecha = anom.index[k]
        antes, despues = f.iloc[:k].mean() * 100, f.iloc[k:].mean() * 100
        fila = {"nombre": e.nombre.strip(), "fecha_cambio": fecha.strftime("%Y-%m"), "p": p,
                "sat_antes_%": antes, "sat_despues_%": despues}
        s = h.serie(e.sid, "td", [10, 11, 12])
        for seg, m in (("antes", s.index < fecha.year + (fecha.month >= 10)),
                       ("despues", s.index > fecha.year + (fecha.month >= 10))):
            ss = s[m]
            fila[f"td_ond_{seg}"] = sen(ss.index.values.astype(float), ss.values) * 10 if len(ss) >= 10 else np.nan
            fila[f"n_{seg}"] = len(ss)
        filas.append(fila)
    r = pd.DataFrame(filas).merge(uso[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    r.to_csv("analisis/27_cambio_sensor.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.sort_values("delta_soja_pp").round(3).to_string(index=False))
    sig = r[(r.p < 0.05) & (r["sat_despues_%"] < r["sat_antes_%"])]
    print(f"\nCaída abrupta de saturados (p<0.05): {len(sig)} de {len(r)}; años: "
          f"{sorted(pd.to_datetime(sig.fecha_cambio).dt.year.tolist())}")
    q = r.dropna(subset=["delta_soja_pp"])
    for seg in ("antes", "despues"):
        qq = q.dropna(subset=[f"td_ond_{seg}"])
        print(f"Δcultivos vs tendencia Td oct-dic {seg} del cambio: ρ {stats.spearmanr(qq.delta_soja_pp, qq[f'td_ond_{seg}'])[0]:+.2f} (n={len(qq)})")


if __name__ == "__main__":
    main()
