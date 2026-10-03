"""Radiosondeos IGRA v2: viento, temperatura y humedad en 925, 850 y 700 hPa.

Formato IGRA v2 (Durre et al. 2006; documentación de NCEI): encabezado que empieza con '#',
luego una línea por nivel con columnas fijas:
  PRESS 10-15 (Pa), GPH 17-21 (m), TEMP 23-27 (°C×10), RH 29-33 (%×10),
  DPDP 35-39 (°C×10), WDIR 41-45 (grados), WSPD 47-51 (m/s×10). Faltante = -9999 / -8888.
Se guardan los sondeos de 00 y 12 UTC (±1 h) de oct-mar 1979-2025.
"""
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

NIVELES = (92500, 85000, 70000)


def num(s, escala=1.0):
    s = s.strip()
    if not s or s.startswith("-9999") or s.startswith("-8888"):
        return np.nan
    v = float(s.rstrip("ABab"))
    return v / escala


def leer(zip_path):
    sid = os.path.basename(zip_path)[:11]
    filas = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as fh:
            texto = io.TextIOWrapper(fh, encoding="ascii", errors="ignore")
            cab = None
            for line in texto:
                if line.startswith("#"):
                    y, m, d, h = int(line[13:17]), int(line[18:20]), int(line[21:23]), int(line[24:26])
                    ok = 1979 <= y <= 2025 and m in (10, 11, 12, 1, 2, 3) and h in (23, 0, 1, 11, 12, 13)
                    cab = (y, m, d, h) if ok else None
                    continue
                if cab is None:
                    continue
                p = num(line[9:15])
                if p not in NIVELES:
                    continue
                y, m, d, h = cab
                filas.append({
                    "fecha": pd.Timestamp(y, m, d) + pd.Timedelta(hours=h), "p": int(p / 100),
                    "t": num(line[22:27], 10), "dpd": num(line[34:39], 10),
                    "wdir": num(line[40:45]), "wspd": num(line[46:51], 10)})
    df = pd.DataFrame(filas)
    # redondear a la hora sinóptica más cercana (00 o 12 UTC)
    df["fecha"] = df.fecha.dt.round("12h")
    df = df.drop_duplicates(["fecha", "p"])
    rad = np.deg2rad(df.wdir)
    df["u"] = -df.wspd * np.sin(rad)
    df["v"] = -df.wspd * np.cos(rad)
    ancho = df.pivot(index="fecha", columns="p", values=["u", "v", "wspd", "t", "dpd"])
    ancho.columns = [f"{a}{b}" for a, b in ancho.columns]
    ancho["sid"] = sid
    return ancho


def main():
    os.makedirs("data/igra/proc", exist_ok=True)
    for z in sorted(glob.glob("data/igra/*-data.txt.zip")):
        a = leer(z)
        a.to_parquet(f"data/igra/proc/{os.path.basename(z)[:11]}.parquet")
        n00 = (a.index.hour == 0).sum()
        n12 = (a.index.hour == 12).sum()
        print(os.path.basename(z)[:11], "sondeos 00 UTC:", n00, "12 UTC:", n12,
              "con v850:", a.get("v850", pd.Series(dtype=float)).notna().sum(), flush=True)


if __name__ == "__main__":
    main()
