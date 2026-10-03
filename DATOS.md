# Data sources

All data are public. Raw and derived data are not included in this repository because of their size; the table lists
where each dataset comes from and where the analysis expects it. After downloading, `./REPRODUCIR.sh` recomputes all
results, figures and the manuscript PDF.

| Dataset | Version / period | Source | Local folder | How |
|---|---|---|---|---|
| HadISD sub-daily station data | v3.4.3.2025f, WMO block 850000–899999 (+ humidity file) | https://www.metoffice.gov.uk/hadobs/hadisd/v343_2025f/download.html | `data/hadisd/` | manual download of `WMO_850000-899999.tar.gz`, `WMO_850000-899999_humidity.tar.gz` and `hadisd_station_fullinfo_v343_2025f.txt` |
| IGRA v2 radiosondes | period of record | https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/ | `data/igra/` | `<station>-data.txt.zip` for the eight stations listed in `analisis/07_igra.py` (profiles use Resistencia ARM00087155, Córdoba ARM00087344, Ezeiza ARM00087576 and Santa Rosa ARM00087623) |
| ERA5 single and pressure levels | Oct–Mar 1979–2025, 10–45°S, 75–45°W | Copernicus Climate Data Store (free account and API key required) | `data/era5/` | `analisis/06_descarga_era5.py`, `06b_descarga_bloques.py`, `06c_descarga_pl_liviana.py` |
| NOAA ISD full hourly reports (fog) | 1980–2025 | https://www.ncei.noaa.gov/data/global-hourly/access/ | `data/isd_full/` | one CSV per station and year |
| NOAA CDR NDVI | v5, AVHRR 1981–2013, VIIRS 2014–2025 | https://www.ncei.noaa.gov/thredds/catalog/cdr/ndvi/catalog.xml | `data/ndvi/` | `analisis/32_ndvi_descarga.py` (server-side subsetting) |
| ESA CCI Soil Moisture | v09.2 COMBINED, daily | https://dap.ceda.ac.uk/neodc/esacci/soil_moisture/data/daily_files/COMBINED/v09.2/ | `data/esacci_sm/` | `analisis/38_humedad_suelo_descarga.py` |
| CHIRPS precipitation | v2.0 monthly, 0.05° | IRI/LDEO Data Library (UCSB CHIRPS v2p0 monthly global) | `data/chirps/chirps_mensual.nc` | regional subset in netCDF |
| MapBiomas Argentina | Collection 3, 1985–2025 | https://storage.googleapis.com/mapbiomas-public/initiatives/argentina/lulc/collection_03/integration/ | `data/mapbiomas/` | `integration-argentina_classification_<year>.tif`, then `analisis/25_mapbiomas.py` |
| MapBiomas Paraguay / Uruguay | Collection 2 / Collection 3 | https://storage.googleapis.com/mapbiomas-public/initiatives/ | `data/mapbiomas_py/`, `data/mapbiomas_uy/` | annual classification (Paraguay) and statistics spreadsheet (Uruguay) |
| GHS-BUILT-S | R2023A, 1975 and 2020, 1 km, Mollweide | https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/ | `data/ghsl/` | `GHS_BUILT_S_E<year>_GLOBE_R2023A_54009_1000` |
| Sown area by department | soybean, maize, sunflower, wheat | https://datos.magyp.gob.ar (series `*-serie-*.csv`) | `data/agro/` | direct CSV download |
| Department polygons | — | https://wms.ign.gob.ar/geoserver/ows (WFS layer `ign:departamento`) | `data/agro/` | GeoJSON |
| ONI | — | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt | `data/indices/` | text file |
| SAM (Marshall, 2003) | — | https://legacy.bas.ac.uk/met/gjma/ | `data/indices/` | text file |
| PDO (ERSST v5) | — | https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat | `data/indices/` | text file |
