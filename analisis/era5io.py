"""Lectura unificada de ERA5 niveles de presión: archivos completos (pl_AAAA_MM*.nc, con u, v, q, t a 700/850/925)
y livianos (pl_uv_AAAA.nc: u, v a 700/850; pl_q_AAAA-AAAA.nc: q a 850/925 a 06 y 12 UTC).
Devuelve un Dataset con dimensiones (time, level, latitude, longitude) y variables u, v, q; donde un archivo no
trae una variable o nivel, queda NaN. Evita duplicados de tiempo quedándose con la primera aparición.
"""
import glob

import xarray as xr


def _normal(ds):
    ds = ds.rename({k: v for k, v in (("valid_time", "time"), ("pressure_level", "level")) if k in ds.dims or k in ds.coords})
    return ds[[v for v in ("u", "v", "q") if v in ds.data_vars]].drop_vars(
        [c for c in ("number", "expver") if c in ds.coords], errors="ignore")


def abrir_pl(patron="data/era5/pl_*.nc"):
    """Concatena en el tiempo cada variable por separado (sin perder archivos) y luego las une."""
    import numpy as np
    abiertos = [_normal(xr.open_dataset(f, chunks={})) for f in sorted(glob.glob(patron))]
    por_var = {}
    for v in ("u", "v", "q"):
        trozos = [ds[v] for ds in abiertos if v in ds]
        if not trozos:
            continue
        da = xr.concat(trozos, dim="time", join="outer", coords="minimal", compat="override")
        _, idx = np.unique(da.time.values, return_index=True)
        por_var[v] = da.isel(time=np.sort(idx)).sortby("time")
    return xr.merge(list(por_var.values()), join="outer")
