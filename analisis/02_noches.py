"""Construye la serie de noches por estación (HadISD v3.4.3, temporada cálida).

Noche = promedio de 06 y 09 UTC (03 y 06 hora local), las dos horas sinópticas con mejor
cobertura en todo el período (ver 01_auditoria.csv). Se exigen ambas horas.
Variables: T, Td, DPD = T - Td, q (g/kg), HR (%), Tw (Davies-Jones, presión estándar a la
altura de la estación), Tw de HadISD (para comparar), viento y dirección a 09 UTC.

Temporada cálida = octubre-marzo; la temporada "2001" es oct-2000 a mar-2001.
"""
import glob
import gzip
import os
import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import termo  # noqa: E402

DIR = "data/hadisd/nc"
SALIDA = os.environ.get("SALIDA_NOCHES", "data/noches")
HORAS = tuple(int(h) for h in os.environ.get("HORAS", "6,9").split(","))


def abrir(sid, sufijo):
    gz = glob.glob(f"{DIR}/*_{sid}{sufijo}.nc.gz")[0]
    nc = gz[:-3]
    if not os.path.exists(nc):
        with gzip.open(gz, "rb") as fi, open(nc, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return xr.open_dataset(nc)


def noches_estacion(sid, elev):
    hum = abrir(sid, "_humidity")
    df = pd.DataFrame({"t": hum.temperatures.values, "td": hum.dewpoints.values,
                       "tw_hadisd": hum.wet_bulb_temperature.values},
                      index=pd.DatetimeIndex(hum.time.values))
    hum.close()
    main = abrir(sid, "")
    viento = pd.DataFrame({"ws": main.windspeeds.values, "wd": main.winddirs.values},
                          index=pd.DatetimeIndex(main.time.values))
    main.close()
    df = df.join(viento)
    df = df[(df.index.minute == 0) & df.index.hour.isin(HORAS)]
    df = df[df.index.month.isin([10, 11, 12, 1, 2, 3])].dropna(subset=["t", "td"])
    df = df[df.td <= df.t + 0.05]          # sobresaturación imposible (Willett 2014)
    p = termo.presion_estandar(elev)
    df["tw"] = termo.tw_davies_jones(df.t.values, df.td.values, p)
    df["q"] = termo.q_desde_td(df.td.values, p)
    df["hr"] = termo.rh_desde_td(df.t.values, df.td.values)
    df["fecha"] = df.index.normalize()
    df["hora"] = df.index.hour
    if df.empty:
        return None
    ancho = df.pivot_table(index="fecha", columns="hora", dropna=False,
                           values=["t", "td", "tw", "q", "hr", "tw_hadisd", "ws", "wd"])
    ancho = ancho.reindex(columns=pd.MultiIndex.from_product(
        [["t", "td", "tw", "q", "hr", "tw_hadisd", "ws", "wd"], list(HORAS)]))
    completas = ancho["t"].notna().all(axis=1)
    ancho = ancho[completas]
    out = pd.DataFrame(index=ancho.index)
    for v in ("t", "td", "tw", "q", "hr", "tw_hadisd"):
        out[v] = ancho[v].mean(axis=1)
    out["dpd"] = out.t - out.td
    out["tw09"] = ancho["tw"][HORAS[-1]]
    out["t09"] = ancho["t"][HORAS[-1]]
    out["ws09"] = ancho["ws"][HORAS[-1]]
    out["wd09"] = ancho["wd"][HORAS[-1]]
    out["temporada"] = np.where(out.index.month >= 10, out.index.year + 1, out.index.year)
    return out


def main():
    os.makedirs(SALIDA, exist_ok=True)
    info = pd.read_fwf("data/hadisd/station_fullinfo.txt", header=None,
                       colspecs=[(0, 12), (13, 43), (43, 51), (51, 60), (60, 68)],
                       names=["sid", "nombre", "lat", "lon", "elev"]).set_index("sid")
    ids = open(os.environ.get("DOMINIO", "data/hadisd/dominio.txt")).read().split()
    for sid in ids:
        out = noches_estacion(sid, info.loc[sid, "elev"])
        if out is None or out.empty:
            print(sid, info.loc[sid, "nombre"], "SIN DATOS", flush=True)
            continue
        out.to_parquet(f"{SALIDA}/{sid}.parquet")
        print(sid, info.loc[sid, "nombre"], len(out), flush=True)


if __name__ == "__main__":
    main()
