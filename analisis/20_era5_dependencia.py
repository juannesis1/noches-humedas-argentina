"""¿Cuánto de las estaciones hay dentro de ERA5 a 2 m? Diagnóstico antes de la prueba OMR.

ERA5 analiza T y HR a 2 m con una interpolación óptima (2D-OI) que usa las observaciones SYNOP
(Hersbach et al. 2020), con análisis en el centro de cada subventana de 6 h (00, 06, 12, 18 UTC).
De noche el 4D-Var no usa la humedad de superficie (Simmons et al. 2010). Si eso es así, a 06 UTC
ERA5 debería pegarse a nuestras estaciones mucho más que a 09 UTC (fuera del análisis 2D-OI).
Métrica: con anomalías diarias (se resta la media mensual de cada serie para quitar el sesgo de
altura y de punto de grilla), correlación y desvío estándar de la diferencia estación − ERA5,
en el punto de grilla más cercano, por hora (06 y 09 UTC) y variable (T, Td).
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

ESTACIONES = os.environ.get("TEND", "analisis/03_tendencias_anomalias_laxo_aj.csv")


def horaria(sid):
    nc = glob.glob(f"data/hadisd/nc/*_{sid}_humidity.nc")[0]
    d = xr.open_dataset(nc)
    df = pd.DataFrame({"t": d.temperatures.values, "td": d.dewpoints.values},
                      index=pd.DatetimeIndex(d.time.values))
    d.close()
    df = df[(df.index.minute == 0) & df.index.hour.isin([6, 9])]
    return df[df.td <= df.t + 0.05].dropna()


def main():
    est = pd.read_csv(ESTACIONES)
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    archivos = sorted(glob.glob("data/era5/sfc_*.nc"))
    ds = xr.open_mfdataset(archivos, combine="by_coords").rename({"valid_time": "time"})
    filas = []
    for _, e in est.iterrows():
        p = ds.sel(latitude=e.lat, longitude=e.lon, method="nearest")[["t2m", "d2m"]].load()
        era = pd.DataFrame({"t": p.t2m.values - 273.15, "td": p.d2m.values - 273.15},
                           index=pd.DatetimeIndex(p.time.values))
        obs = horaria(e.sid)
        for h in (6, 9):
            o = obs[obs.index.hour == h]
            r = era[era.index.hour == h]
            m = o.join(r, lsuffix="_o", rsuffix="_e", how="inner")
            if len(m) < 300:
                continue
            for v in ("t", "td"):
                a = m[f"{v}_o"] - m[f"{v}_o"].groupby([m.index.year, m.index.month]).transform("mean")
                b = m[f"{v}_e"] - m[f"{v}_e"].groupby([m.index.year, m.index.month]).transform("mean")
                filas.append({"nombre": e.nombre.strip(), "hora": h, "var": v, "n": len(m),
                              "r": np.corrcoef(a, b)[0, 1], "sd_dif": (a - b).std(),
                              "sesgo": (m[f"{v}_o"] - m[f"{v}_e"]).mean()})
    res = pd.DataFrame(filas)
    res.to_csv("analisis/20_era5_dependencia.csv", index=False)
    piv = res.pivot_table(index="nombre", columns=["var", "hora"], values=["r", "sd_dif"])
    pd.set_option("display.width", 250)
    print(piv.round(2).to_string())
    print("\nMedianas:")
    print(res.groupby(["var", "hora"])[["r", "sd_dif", "sesgo"]].median().round(3))
    print(f"\nMeses ERA5 usados: {len(archivos)}")


if __name__ == "__main__":
    main()
