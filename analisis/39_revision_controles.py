"""Controles pedidos por la revisión "en frío" (revision_estricta.md, sección I; 3 de octubre).

I-7  Índice de intensificación (n = 16): p con n efectivo de Clifford–Dutilleul, además de los sustitutos.
I-B  NDVI AVHRR: la deriva orbital depende de latitud y estación. ¿La asociación NDVI~cultivos sobrevive al control
     por latitud, dentro de la Pampa y solo con AVHRR (1982-2013)?
I-D  Enfriamiento nocturno vs expansión: ¿sobrevive al control por crecimiento urbano?
I-E  Descomposición con sondeos de Resistencia: cobertura de noches tipificadas por período y sesgo de submuestra
     (tendencia de Td en noches con sondeo vs todas).
I-F  Test dentro de la Pampa (lat < 30.5°S, lon < 58.3°W): cultivos e intensificación vs Td oct-dic.
Salida: analisis/39_resumen.txt
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

nul = importlib.import_module("24_nulo_espacial")
p36 = importlib.import_module("36_ndvi_panel")
L = []


def pampa(df):
    return df[(df.lat < -30.5) & (df.lon < -58.3)]


def p_neff(x, y, lat, lon):
    D = nul.dist(np.asarray(lat), np.asarray(lon))
    xr_, yr = stats.rankdata(x).astype(float), stats.rankdata(y).astype(float)
    rho = stats.spearmanr(x, y)[0]
    ne = nul.n_eff(xr_, yr, D)
    t = rho * np.sqrt((ne - 2) / (1 - rho ** 2))
    return rho, ne, 2 * stats.t.sf(abs(t), ne - 2)


def main():
    # I-7 y I-F: intensificación
    it = pd.read_csv("analisis/26_intensificacion.csv").dropna(subset=["d_int", "td_ond"])
    rho, ne, p = p_neff(it.d_int, it.td_ond, it.lat, it.lon)
    L.append(f"I-7 Intensificación vs Td oct-dic: ρ {rho:+.2f}, n {len(it)}, n_eff {ne:.1f}, p(n_eff) {p:.3f}")
    u = pd.read_csv("analisis/18_urbano.csv").dropna(subset=["delta_soja_pp"])
    for nom, df, col in (("cultivos", u, "delta_soja_pp"), ("intensificación", it, "d_int")):
        q = pampa(df)
        r = stats.spearmanr(q[col], q.td_ond)
        L.append(f"I-F Pampa: {nom} vs Td oct-dic: ρ {r[0]:+.2f} (p {r[1]:.3f}, n {len(q)})")
    # I-B: NDVI
    nd = pd.read_csv("analisis/35_ndvi_series.csv")
    est = u[["nombre", "lat", "lon", "delta_soja_pp"]].copy()
    est["nombre"] = est.nombre.str.strip()
    for etiqueta, fin in (("1982-2025", 2025), ("1982-2013 (AVHRR)", 2013)):
        for meses, vn in (([10, 11, 12], "oct-dic"), ([1, 2, 3], "ene-mar")):
            g = nd[nd.mes.isin(meses) & (nd.temporada >= 1982) & (nd.temporada <= fin)]
            g = g.groupby(["nombre", "temporada"]).ndvi.mean().reset_index()
            t = g.groupby("nombre").apply(lambda h: sen(h.temporada.values.astype(float), h.ndvi.values) * 10,
                                          include_groups=False).rename("tend").reset_index()
            t["nombre"] = t.nombre.str.strip()
            t = t.merge(est, on="nombre").dropna()
            r_lat = p36.parcial(t.delta_soja_pp.values, t.tend.values, np.c_[t.lat])[0]
            r_ll = p36.parcial(t.delta_soja_pp.values, t.tend.values, np.c_[t.lat, t.lon])[0]
            r_lat_ndvi = stats.spearmanr(t.lat, t.tend)[0]
            q = pampa(t)
            rp = stats.spearmanr(q.delta_soja_pp, q.tend)
            L.append(f"I-B NDVI {vn} {etiqueta}: ρ(cultivos) {stats.spearmanr(t.delta_soja_pp, t.tend)[0]:+.2f}; "
                     f"parcial|lat {r_lat:+.2f}; parcial|lat,lon {r_ll:+.2f}; ρ(lat, NDVI) {r_lat_ndvi:+.2f}; "
                     f"Pampa ρ {rp[0]:+.2f} (p {rp[1]:.3f}, n {len(q)})")
    # I-D: enfriamiento
    v = pd.read_csv("analisis/30_viento_nubes.csv").assign(nombre=lambda x: x.nombre.str.strip())
    v = v.merge(u.assign(nombre=u.nombre.str.strip())[["nombre", "db_10km"]], on="nombre").dropna(
        subset=["enfriamiento_tend", "delta_soja_pp"])
    r0 = stats.spearmanr(v.delta_soja_pp, v.enfriamiento_tend)[0]
    r1 = p36.parcial(v.delta_soja_pp.values, v.enfriamiento_tend.values, np.c_[v.db_10km])
    r2 = p36.parcial(v.delta_soja_pp.values, v.enfriamiento_tend.values, np.c_[v.db_10km, v.lat, v.lon])
    ru = stats.spearmanr(v.db_10km, v.enfriamiento_tend)[0]
    L.append(f"I-D Enfriamiento vs cultivos: ρ {r0:+.2f}; parcial|urbano {r1[0]:+.2f} (p {r1[1]:.3f}); "
             f"parcial|urbano,lat,lon {r2[0]:+.2f} (p {r2[1]:.3f}); ρ(urbano, enfriamiento) {ru:+.2f}; n {len(v)}")
    # I-E: cobertura de sondeos y sesgo de submuestra
    b12 = importlib.import_module("12b_descomposicion_igra")
    tend3 = importlib.import_module("03_tendencias")
    tipo = b12.tipos_igra()
    tr = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    tr = tr[(tr.temporada == "ONDJFM") & tr.incluida & ~tr.nombre.str.contains("CERES")]
    cob, filas = [], []
    for _, e in tr.iterrows():
        d = pd.read_parquet(f"data/noches_ajustadas_pares_laxo/{e.sid}.parquet")
        d = tend3.anomalias(d[d.index.month.isin([10, 11, 12])])
        d = d.dropna(subset=["td"])
        temp = np.where(d.index.month >= 10, d.index.year + 1, d.index.year)
        con = tipo.reindex(d.index).notna().values
        cob.append(pd.DataFrame({"p": np.where(temp <= 2002, "1980-2002", "2003-2025"), "con": con}))
        s_all = pd.Series(d.td.values, index=temp).groupby(level=0).mean()
        s_sub = pd.Series(d.td.values[con], index=temp[con]).groupby(level=0).mean()
        s_all, s_sub = s_all[(s_all.index >= 1980)], s_sub[(s_sub.index >= 1980)]
        filas.append({"nombre": e.nombre.strip(), "todas": sen(s_all.index.values.astype(float), s_all.values) * 10,
                      "con_sondeo": sen(s_sub.index.values.astype(float), s_sub.values) * 10})
    c = pd.concat(cob).groupby("p").con.mean()
    f = pd.DataFrame(filas).merge(u.assign(nombre=u.nombre.str.strip())[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    q = f.dropna(subset=["delta_soja_pp"])
    L.append(f"I-E Noches oct-dic con sondeo de Resistencia: 1980-2002 {c.iloc[0]:.0%}, 2003-2025 {c.iloc[1]:.0%}")
    L.append(f"I-E Tendencia Td oct-dic (mediana): todas {f.todas.median():+.3f}, solo noches con sondeo "
             f"{f.con_sondeo.median():+.3f}; r entre estaciones {stats.pearsonr(f.todas, f.con_sondeo)[0]:.2f}; "
             f"ρ(cultivos) todas {stats.spearmanr(q.delta_soja_pp, q.todas)[0]:+.2f}, con sondeo "
             f"{stats.spearmanr(q.delta_soja_pp, q.con_sondeo)[0]:+.2f}")
    open("analisis/39_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
