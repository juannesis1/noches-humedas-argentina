"""Explicaciones alternativas físicas del secado nocturno: viento, nubosidad y enfriamiento nocturno.

Si las noches de la zona agrícola se volvieron más calmas, más despejadas o con más enfriamiento radiativo,
podría depositarse más rocío y bajar la Td al amanecer sin que cambie la evapotranspiración. Para cada
estación (oct-dic, datos crudos de HadISD, 1980-2025):
  - velocidad del viento media de 06 y 09 UTC;
  - nubosidad total y baja (octas) media de 06 y 09 UTC;
  - enfriamiento nocturno: T(21 UTC, 18 local) − T(09 UTC, 06 local) del mismo ciclo (atardecer → amanecer).
Tendencias de Sen sobre anomalías por temporada (clim. 1981-2010), relación con la expansión de cultivos y con
la tendencia de Td; y correlación parcial Td ~ cultivos controlando lat, lon y cada una de estas tendencias.
Advertencia: el viento tiene inhomogeneidades de anemómetro (calmas −7 pp/déc) y la nubosidad la observa una
persona (cobertura baja en los 2000 en algunas estaciones).
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

MESES = [10, 11, 12]


def estacional(s, min_dias=40):
    s = s[s.index.month.isin(MESES)].dropna()
    temp = s.index.year + (s.index.month >= 10)
    base = s[(temp >= 1981) & (temp <= 2010)]
    if len(base) < 300:
        return pd.Series(dtype=float)
    clim = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True, min_periods=10).mean().iloc[15:-15]
    clim.index = range(1, 367)
    a = s - clim.loc[s.index.dayofyear].values
    g = pd.Series(a.values, index=temp).groupby(level=0)
    out = g.mean()[g.size() >= min_dias]
    return out[(out.index >= 1980) & (out.index <= 2025)]


def tendencia(s):
    ok = len(s) >= 30 and (s.index <= 1989).sum() >= 5 and (s.index >= 2016).sum() >= 5
    return sen(s.index.values.astype(float), s.values) * 10 if ok else np.nan


def parcial(x, y, Z):
    X = np.c_[np.ones(len(x)), Z]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)[0]


def main():
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    td_ond = pd.read_csv("analisis/18_urbano.csv").rename(columns={"Unnamed: 0": "nombre"})
    td_ond = td_ond.set_index(td_ond.columns[0]).td_ond if "td_ond" in td_ond else None
    filas = []
    for _, e in est.iterrows():
        main_nc = glob.glob(f"data/hadisd/nc/*_{e.sid}.nc")[0]
        d = xr.open_dataset(main_nc)
        df = pd.DataFrame({"t": d.temperatures.values, "ws": d.windspeeds.values,
                           "nt": d.total_cloud_cover.values, "nb": d.low_cloud_cover.values},
                          index=pd.DatetimeIndex(d.time.values))
        d.close()
        df = df[df.index.minute == 0]
        # valores de relleno y códigos fuera de rango físico → faltante (HadISD usa rellenos tipo −2e30 / 99)
        df["t"] = df.t.where(df.t.between(-40, 55))
        df["ws"] = df.ws.where(df.ws.between(0, 60))
        df["nt"] = df.nt.where(df.nt.between(0, 8))
        df["nb"] = df.nb.where(df.nb.between(0, 8))
        noc = df[df.index.hour.isin([6, 9])]
        dia = lambda v: noc[v].groupby(noc.index.normalize()).mean()
        t21 = df.t[df.index.hour == 21]
        t09 = df.t[df.index.hour == 9]
        t21.index = (t21.index + pd.Timedelta(hours=12)).normalize()       # 21 UTC de d → noche que termina d+1
        t09.index = t09.index.normalize()
        enfr = (t21 - t09.reindex(t21.index)).dropna()
        enfr = enfr[~enfr.index.duplicated()]
        fila = {"nombre": e.nombre.strip(), "lat": e.lat, "lon": e.lon}
        for nombre, s in (("viento", dia("ws")), ("nub_total", dia("nt")), ("nub_baja", dia("nb")), ("enfriamiento", enfr)):
            serie = estacional(s)
            fila[f"{nombre}_tend"] = tendencia(serie)
            fila[f"{nombre}_n"] = len(serie)
        filas.append(fila)
    r = pd.DataFrame(filas).merge(uso[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    if td_ond is not None:
        r["td_ond"] = r.nombre.map(td_ond)
    r.to_csv("analisis/30_viento_nubes.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.sort_values("delta_soja_pp").round(3).to_string(index=False))
    q = r.dropna(subset=["delta_soja_pp", "td_ond"])
    print(f"\nAlta expansión (>20 pp) vs baja (<5 pp), medias de tendencia por década:")
    for c, u in (("viento", "m/s"), ("nub_total", "octas"), ("nub_baja", "octas"), ("enfriamiento", "°C")):
        a = q[q.delta_soja_pp > 20][f"{c}_tend"].mean()
        b = q[q.delta_soja_pp < 5][f"{c}_tend"].mean()
        qq = q.dropna(subset=[f"{c}_tend"])
        print(f"  {c:13s} alta {a:+.3f} baja {b:+.3f} {u} | ρ con cultivos {stats.spearmanr(qq.delta_soja_pp, qq[f'{c}_tend'])[0]:+.2f} "
              f"| ρ con Td oct-dic {stats.spearmanr(qq.td_ond, qq[f'{c}_tend'])[0]:+.2f} | "
              f"Td~cultivos parcial (lat, lon, {c}) {parcial(qq.delta_soja_pp.values, qq.td_ond.values, np.c_[qq.lat, qq.lon, qq[f'{c}_tend']]):+.2f} (n={len(qq)})")
    print(f"  referencia: Td~cultivos parcial (lat, lon) {parcial(q.delta_soja_pp.values, q.td_ond.values, np.c_[q.lat, q.lon]):+.2f}")


if __name__ == "__main__":
    main()
