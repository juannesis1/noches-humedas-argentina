"""Auditoría de disponibilidad horaria en HadISD (temporada cálida ONDJFM).

Para cada estación y década, fracción de noches con T y Td válidas en cada hora UTC
nocturna (21 a 09 UTC = 18 a 06 hora local). Sirve para elegir qué horas definen la
métrica nocturna de forma consistente en todo el período (Simpson et al. 2023; Dai 2006).
"""
import glob
import gzip
import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr

DIR = "data/hadisd/nc"
HORAS = [21, 0, 3, 6, 9]


def abrir(sid):
    gz = glob.glob(f"{DIR}/*_{sid}_humidity.nc.gz")[0]
    nc = gz[:-3]
    if not os.path.exists(nc):
        with gzip.open(gz, "rb") as fi, open(nc, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return xr.open_dataset(nc)


def main():
    info = pd.read_fwf("data/hadisd/station_fullinfo.txt", header=None,
                       colspecs=[(0, 12), (13, 43), (43, 51), (51, 60), (60, 68)],
                       names=["sid", "nombre", "lat", "lon", "elev"]).set_index("sid")
    ids = open("data/hadisd/dominio.txt").read().split()
    filas = []
    for sid in ids:
        ds = abrir(sid)
        df = pd.DataFrame({"t": ds.temperatures.values, "td": ds.dewpoints.values},
                          index=pd.DatetimeIndex(ds.time.values))
        ds.close()
        df = df.dropna()
        df = df[df.index.month.isin([10, 11, 12, 1, 2, 3])]
        df = df[(df.index.year >= 1971) & (df.index.year <= 2025)]
        df["hora"] = df.index.hour
        df["min"] = df.index.minute
        df = df[(df["min"] == 0) & df.hora.isin(HORAS)]
        # la "noche" de 21 UTC pertenece a la fecha siguiente
        fecha = df.index.normalize() + pd.to_timedelta((df.hora == 21).astype(int), unit="D")
        df["fecha"] = fecha
        df["decada"] = (df.fecha.dt.year // 10) * 10
        dias = pd.Series(1, index=pd.date_range("1971-01-01", "2025-12-31"))
        dias = dias[dias.index.month.isin([10, 11, 12, 1, 2, 3])]
        n_dec = dias.groupby((dias.index.year // 10) * 10).size()
        for dec, g in df.groupby("decada"):
            fila = {"sid": sid, "nombre": info.loc[sid, "nombre"], "lat": info.loc[sid, "lat"],
                    "lon": info.loc[sid, "lon"], "elev": info.loc[sid, "elev"], "decada": dec}
            for h in HORAS:
                fila[f"h{h:02d}"] = round(g[g.hora == h].fecha.nunique() / n_dec[dec], 2)
            filas.append(fila)
        print(sid, info.loc[sid, "nombre"], flush=True)
    res = pd.DataFrame(filas)
    res.to_csv("analisis/01_auditoria.csv", index=False)


if __name__ == "__main__":
    main()
