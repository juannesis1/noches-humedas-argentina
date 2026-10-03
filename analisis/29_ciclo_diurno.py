"""Ciclo diurno de la huella agrícola: ¿a qué hora del día aparece el secado?

Si el secado viene de la superficie y lo retiene la capa estable nocturna, debería ser máximo de noche y al
amanecer y debilitarse por la tarde, cuando la capa de mezcla diluye la señal en 1-2 km. Si fuera regional
(advección de aire más seco), aparecería a todas las horas.
Para cada una de las 8 horas sinópticas (00-21 UTC; 21-18 hora local): Td de oct-dic de esa hora, anomalía
respecto de la climatología diaria 1981-2010 (media móvil de 31 días), temporada válida con ≥ 50 % de días
y ≥ 9 por mes, tendencia de Sen 1980-2025 (≥ 30 temporadas, ≥ 5 en cada extremo). Datos CRUDOS de HadISD (la
homogeneización se estimó para la media 06-09 UTC); como referencia se repite 06+09 UTC crudo.
Salida: cobertura por hora y década, y ρ (Spearman) entre la expansión de cultivos y la tendencia por hora.
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

HORAS = [0, 3, 6, 9, 12, 15, 18, 21]
MESES = [int(m) for m in os.environ.get("MESES", "10,11,12").split(",")]
SUF = os.environ.get("SUFIJO", "")


def serie_hora(df):
    """df: Td diaria de una hora (índice fecha). Devuelve anomalía media por temporada válida."""
    temp = df.index.year + (df.index.month >= 10)          # oct-dic de y → temporada y+1
    base = df[(temp >= 1981) & (temp <= 2010)]
    clim = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True, min_periods=10).mean()
    clim = clim.iloc[15:-15]
    clim.index = range(1, 367)
    a = df - clim.loc[df.index.dayofyear].values
    g = pd.DataFrame({"a": a.values, "t": temp, "m": df.index.month})
    n = g.groupby("t").size()
    pm = g.groupby(["t", "m"]).size().unstack().reindex(columns=MESES).fillna(0)
    ok = (n >= 0.5 * 30.5 * len(MESES)) & (pm >= 9).all(axis=1)
    s = g.groupby("t").a.mean()[ok]
    return s[(s.index >= 1980) & (s.index <= 2025)]


def main():
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    filas, cob = [], []
    for _, e in est.iterrows():
        nc = glob.glob(f"data/hadisd/nc/*_{e.sid}_humidity.nc")[0]
        d = xr.open_dataset(nc)
        df = pd.DataFrame({"t": d.temperatures.values, "td": d.dewpoints.values}, index=pd.DatetimeIndex(d.time.values))
        d.close()
        df = df[(df.index.minute == 0) & df.index.month.isin(MESES) & (df.index.year >= 1979)].dropna()
        df = df[df.td <= df.t + 0.05]
        for h in HORAS + ["06+09"]:
            if h == "06+09":
                x = df[df.index.hour.isin([6, 9])]
                x = x.groupby(x.index.normalize()).filter(lambda g: len(g) == 2) if False else x
                td = x.td.groupby(x.index.normalize()).mean()
            else:
                x = df[df.index.hour == h]
                td = pd.Series(x.td.values, index=x.index.normalize())
                td = td[~td.index.duplicated()]
            dec = (td.index.year // 10) * 10
            for k, v in td.groupby(dec).size().items():
                cob.append({"nombre": e.nombre.strip(), "hora": str(h), "decada": k, "frac": v / (30.5 * len(MESES) * 10)})
            s = serie_hora(td)
            ok = len(s) >= 30 and (s.index <= 1989).sum() >= 5 and (s.index >= 2016).sum() >= 5
            filas.append({"nombre": e.nombre.strip(), "hora": str(h), "n": len(s),
                          "td_sen": sen(s.index.values.astype(float), s.values) * 10 if ok else np.nan})
    r = pd.DataFrame(filas).merge(uso[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    r = r.merge(est.assign(nombre=est.nombre.str.strip())[["nombre", "lat", "lon"]], on="nombre", how="left")
    r.to_csv(f"analisis/29_ciclo_diurno{SUF}.csv", index=False)
    c = pd.DataFrame(cob).pivot_table(index="hora", columns="decada", values="frac", aggfunc="median")
    print("Cobertura mediana (fracción de días oct-dic con dato) por hora y década:")
    print(c.round(2).to_string())
    print("\nρ(expansión de cultivos, tendencia de Td oct-dic) por hora (UTC; local = UTC − 3):")
    for h in [str(x) for x in HORAS] + ["06+09"]:
        q = r[(r.hora == h)].dropna(subset=["td_sen", "delta_soja_pp"])
        if len(q) < 10:
            print(f"  {h:>5}: n={len(q)} insuficiente")
            continue
        alta = q[q.delta_soja_pp > 20].td_sen.mean()
        baja = q[q.delta_soja_pp < 5].td_sen.mean()
        print(f"  {h:>5}: ρ {stats.spearmanr(q.delta_soja_pp, q.td_sen)[0]:+.2f} (n={len(q)})  "
              f"Td alta exp {alta:+.2f}, baja {baja:+.2f}, contraste {alta - baja:+.2f} °C/déc")


if __name__ == "__main__":
    main()
