"""Descarga del NDVI diario del NOAA CDR (AVHRR/VIIRS, v5; Vermote et al., doi:10.7289/V5ZG6QH9), recortado a la
región (40-23°S, 68-53°W) y a 0.1° (horizStride=2) con el servicio NCSS de THREDDS de NCEI. Meses oct-mar, 1981-2025.
Genera la lista de URLs desde los catálogos anuales; la descarga en paralelo la hace xargs (ver log).
"""
import re
import sys
import urllib.request

MESES = {"01", "02", "03", "10", "11", "12"}
for y in range(1981, 2026):
    cat = urllib.request.urlopen(f"https://www.ncei.noaa.gov/thredds/catalog/cdr/ndvi/{y}/catalog.xml", timeout=60).read().decode()
    for path in re.findall(r'urlPath="([^"]+\.nc)"', cat):
        fecha = re.search(r"_(\d{8})_c", path).group(1)
        if fecha[4:6] in MESES:
            url = (f"https://www.ncei.noaa.gov/thredds/ncss/grid/{path}?var=NDVI&north=-23&south=-40"
                   f"&west=-68&east=-53&horizStride=2&accept=netcdf")
            print(url, f"data/ndvi/ndvi_{fecha}.nc")
    sys.stderr.write(f"{y} ")
