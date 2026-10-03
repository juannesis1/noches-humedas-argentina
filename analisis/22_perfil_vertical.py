"""Estructura vertical del secado con radiosondeos (independiente de ERA5).

Si el secado de oct-dic es de origen superficial (menos evapotranspiración), debería ser máximo
cerca del suelo y debilitarse con la altura; si viene de la circulación (menos transporte de
humedad), debería verse también en la capa del jet (~500-1500 m sobre el suelo).
IGRA v2 (Durre et al. 2006): todos los niveles con altura geopotencial, T y depresión del punto
de rocío. La superficie es el nivel con LVLTYP2 = 1 (columna 2). q (g/kg) se interpola
linealmente en ln(p) a 0, 25, 50, 100, 150 y 200 hPa por encima de la presión de superficie
(≈ 0, 230, 470, 950, 1450 y 2000 m). Se usa presión y no altura porque desde ~2005 los sondeos
argentinos llegan por GTS sin altura geopotencial en los niveles significativos. Se exige un
nivel válido a menos de 40 hPa por encima y por debajo de cada nivel (salvo la superficie).
12 UTC = 09 hora local (todavía con capa estable nocturna en buena parte de la temporada).
Tendencias de Sen de las anomalías por temporada (≥ 25 sondeos válidos por ventana).
COMUN=1: misma muestra en todos los niveles (solo sondeos con q válida de 0 a 100 hPa), para no
comparar niveles con distinta cobertura temporal (sin esto, en Córdoba 50 y 150 hPa empiezan en
1992-93 y la superficie en 1980). Es la versión que va al paper. Mínimo 15 sondeos por temporada.
Advertencia: el nivel de superficie del sondeo es la observación de la estación (no la sonda).
Advertencia: cambios de sensor de humedad (p. ej. Vaisala RS80 → RS92 → RS41) pueden introducir
saltos; por eso se compara con la tendencia de superficie de HadISD de la misma estación.
"""
import glob
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import mk_hamed_rao, sen  # noqa: E402
import termo  # noqa: E402

ALTURAS = (0, 25, 50, 100, 150, 200)      # hPa sobre la superficie
COMUN = os.environ.get("COMUN", "1") == "1"
MIN_SONDEOS = 15 if COMUN else 25
VENTANAS = {"OND": [10, 11, 12], "JFM": [1, 2, 3]}
NOMBRES = {"87344": "Córdoba", "87155": "Resistencia", "87576": "Ezeiza", "87623": "Santa Rosa",
           "87047": "Salta", "83827": "Foz do Iguaçu", "83928": "Uruguaiana", "86218": "Asunción"}


def num(s, escala=1.0):
    s = s.strip().rstrip("ABab")
    if not s or s.startswith("-9999") or s.startswith("-8888"):
        return np.nan
    return float(s) / escala


def perfiles(zip_path):
    filas, niveles, cab = [], [], None

    def cerrar():
        if cab is None or not niveles:
            return
        n = pd.DataFrame(niveles, columns=["sup", "p", "z", "t", "dpd"]).dropna(subset=["p", "t", "dpd"])
        s = n[n.sup]
        if s.empty:
            return
        p0 = s.p.iloc[0]
        n = n[n.p <= p0].copy()
        n["agl"] = p0 - n.p                     # hPa por encima de la superficie
        n["q"] = termo.q_desde_td((n.t - n.dpd).values, n.p.values)
        n = n.sort_values("agl").drop_duplicates("agl")
        fila = {"fecha": cab}
        for h in ALTURAS:
            if h == 0:
                fila["q0"] = s.pipe(lambda x: termo.q_desde_td((x.t - x.dpd).values, x.p.values))[0]
                fila["t0"] = s.t.iloc[0]
                continue
            abajo, arriba = n[n.agl <= h], n[n.agl >= h]
            if abajo.empty or arriba.empty:
                continue
            a, b = abajo.iloc[-1], arriba.iloc[0]
            if h - a.agl > 40 or b.agl - h > 40:
                continue
            la, lb, lh = np.log(p0 - a.agl), np.log(p0 - b.agl), np.log(p0 - h)
            w = 0.0 if b.agl == a.agl else (lh - la) / (lb - la)
            fila[f"q{h}"] = a.q + w * (b.q - a.q)
            fila[f"t{h}"] = a.t + w * (b.t - a.t)
        filas.append(fila)

    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as fh:
            for line in io.TextIOWrapper(fh, encoding="ascii", errors="ignore"):
                if line.startswith("#"):
                    cerrar()
                    niveles = []
                    y, m, d, h = int(line[13:17]), int(line[18:20]), int(line[21:23]), int(line[24:26])
                    ok = 1979 <= y <= 2025 and m in (10, 11, 12, 1, 2, 3) and h in (11, 12, 13)
                    cab = pd.Timestamp(y, m, d, 12) if ok else None
                    continue
                if cab is None:
                    continue
                p = num(line[9:15], 100)
                niveles.append((line[1] == "1", p, num(line[16:21]), num(line[22:27], 10),
                                num(line[34:39], 10)))
            cerrar()
    return pd.DataFrame(filas).drop_duplicates("fecha").set_index("fecha")


def main():
    os.makedirs("data/igra/perfil", exist_ok=True)
    filas = []
    for zp in sorted(glob.glob("data/igra/*-data.txt.zip")):
        sid = os.path.basename(zp)[:11]
        dest = f"data/igra/perfil/{sid}.parquet"
        d = pd.read_parquet(dest) if os.path.exists(dest) else perfiles(zp)
        d.to_parquet(dest)
        nombre = NOMBRES.get(sid[-5:], sid)
        d["temporada"] = np.where(d.index.month >= 10, d.index.year + 1, d.index.year)
        for vn, meses in VENTANAS.items():
            dd = d[d.index.month.isin(meses) & (d.temporada >= 1980) & (d.temporada <= 2025)]
            if COMUN:
                dd = dd.dropna(subset=["q0", "q25", "q50", "q100"])
            for h in ALTURAS:
                for var in ("q", "t"):
                    c = f"{var}{h}"
                    if c not in dd:
                        continue
                    x = dd[c]
                    base = x[(dd.temporada >= 1981) & (dd.temporada <= 2010)]
                    clim = base.groupby(base.index.month).mean()
                    a = x - clim.reindex(x.index.month).values
                    g = a.groupby(dd.temporada)
                    s = g.mean()[g.count() >= MIN_SONDEOS]
                    if len(s) < 20:
                        continue
                    xs = s.index.values.astype(float)
                    filas.append({"estacion": nombre, "ventana": vn, "altura": h, "var": var,
                                  "n_temp": len(s), "sen_dec": sen(xs, s.values) * 10,
                                  "p": mk_hamed_rao(xs, s.values)[1],
                                  "primera": int(s.index.min()), "ultima": int(s.index.max())})
        print(sid, nombre, len(d), "sondeos 12 UTC oct-mar", flush=True)
    res = pd.DataFrame(filas)
    res.to_csv(f"analisis/22_perfil_vertical{'_comun' if COMUN else ''}.csv", index=False)
    pd.set_option("display.width", 250)
    for var, u in (("q", "g/kg/déc"), ("t", "°C/déc")):
        for vn in VENTANAS:
            t = res[(res["var"] == var) & (res.ventana == vn)].pivot(index="estacion", columns="altura",
                                                                     values="sen_dec")
            n = res[(res["var"] == var) & (res.ventana == vn)].pivot(index="estacion", columns="altura",
                                                                     values="n_temp")
            print(f"\n{var} {vn} ({u}); columnas = hPa sobre la superficie")
            print(t.round(3).to_string())
            print("  temporadas:", n.min(axis=1).astype("Int64").to_dict())


if __name__ == "__main__":
    main()
