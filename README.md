# gta6-trophy-rarity-forecast-01

[![make.fun](https://img.shields.io/badge/make.fun-gta6--trophy--rarity--forecast--01-blue)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![forecast](https://img.shields.io/badge/type-forecasting-green)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![prize](https://img.shields.io/badge/prize-%24250_USDC-gold)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy: it decomposes each rate into earners over tracked players, fits log-earners and log-population against log-hours since release by recency-weighted least squares with tier shrinkage, and falls back to a dilution-shifted last rate with a wide spread when the history is flat, so a refresh of Exophase's counts cannot catch the forecast outside its band.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Enter the competition at [make.fun](https://make.fun/?ref=ye9vf8mj).
