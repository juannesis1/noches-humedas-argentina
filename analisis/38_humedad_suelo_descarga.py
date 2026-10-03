"""Humedad del suelo satelital ESA CCI SM v09.2 COMBINED (diaria, 0.25°, 1978–) para oct-dic, 1980-2025.

Motivo (revisión I-A, 3 de octubre): la lluvia CHIRPS en ±0.5° mide con mucho error la humedad del suelo, así que la
parcial "lluvia | NDVI" subestima su papel. La humedad superficial del suelo es el control directo.
Cada archivo diario (~1-5 MB, global) se baja, se recorta a 18-42°S, 70-48°W, se acumula en la media mensual y se borra.
Salida: data/esacci_sm/sm_mensual.nc (sm medio y n de días válidos por celda y mes). Reanudable: guarda cada mes.
Uso: python analisis/38_humedad_suelo_descarga.py   (log en analisis/logs_pendientes/38_sm.log)
"""
import glob
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

URL = "https://dap.ceda.ac.uk/neodc/esacci/soil_moisture/data/daily_files/COMBINED/v09.2/{y}/" \
      "ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-{d}000000-fv09.2.nc"
SAL = "data/esacci_sm"
MESES = [10, 11, 12]


def dia(fecha):
    url = URL.format(y=fecha.year, d=f"{fecha:%Y%m%d}")
    for i in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                b = r.read()
            tmp = f"{SAL}/tmp_{fecha:%Y%m%d}.nc"
            open(tmp, "wb").write(b)
            with xr.open_dataset(tmp, engine="netcdf4") as ds:
                c = ds.sm.sel(lat=slice(-18, -42), lon=slice(-70, -48)).isel(time=0).load()
            os.remove(tmp)
            return c
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(20 * (i + 1))
        except Exception as e:
            print(time.strftime("%H:%M"), f"reintento {fecha:%Y-%m-%d}: {str(e)[:100]}", flush=True)
            time.sleep(20 * (i + 1))
    return None


def main():
    os.makedirs(SAL, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        for y in range(1979, 2026):
            for m in MESES:
                dest = f"{SAL}/sm_{y}_{m:02d}.nc"
                if os.path.exists(dest):
                    continue
                dias = pd.date_range(f"{y}-{m:02d}-01", periods=pd.Timestamp(y, m, 1).days_in_month)
                campos = [c for c in ex.map(dia, dias) if c is not None]
                if not campos:
                    print(time.strftime("%H:%M"), "sin datos", y, m, flush=True)
                    continue
                st = xr.concat(campos, "dia")
                out = xr.Dataset({"sm": st.mean("dia"), "n": st.notnull().sum("dia")}).expand_dims(
                    time=[pd.Timestamp(y, m, 1)])
                out.to_netcdf(dest)
                print(time.strftime("%H:%M"), "ok", y, m, len(campos), "días", flush=True)
    xr.open_mfdataset(sorted(glob.glob(f"{SAL}/sm_*_*.nc")), combine="by_coords").load().to_netcdf(f"{SAL}/sm_mensual.nc")
    print("listo", flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
