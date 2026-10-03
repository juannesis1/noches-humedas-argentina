# Nocturnal humidity in subtropical Argentina, 1980–2025: analysis code

Code for the study *"Warmer but not more humid: summer nights in subtropical Argentina, 1980–2025, and their
association with urban growth and agricultural intensification"* (Juan Nesis, ORCID 0009-0008-2439-3378).
The manuscript, figures and result tables will be added when the paper is submitted.

## Contents

- `analisis/`: all processing and analysis scripts (Python). `estadistica.py` (Sen slope, Mann–Kendall with the
  Hamed–Rao correction, false discovery rate), `termo.py` (wet-bulb temperature and humidity variables) and `era5io.py`
  (ERA5 reader) are shared modules. Download scripts: `06*` (ERA5), `32` (NDVI), `38` (soil moisture).
- `REPRODUCIR.sh`: runs the whole analysis in order and rebuilds figures and PDFs.
- `DATOS.md`: public data sources and where each dataset is expected.

## Reproducing the results

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# download the data listed in DATOS.md into data/
./REPRODUCIR.sh
```

Each step writes a log to `logs/`. `./REPRODUCIR.sh desde 12` resumes from the first step whose name starts with `12`.

## License

Code: MIT License. The data belong to their providers and are subject to their own terms.
