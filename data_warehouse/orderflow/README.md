# gen.Flow — order-flow data layer

Warstwa danych mikrostruktury (agresja/absorpcja/CVD) — to, czego drzewa nie
widzą z OHLCV. Pluggable: aktywny **Binance aggTrades (darmowy)**, gotowe do
podpięcia **mmt.gg** ($199/mc) i kolejne.

## Pliki
- `providers.py` — abstrakcja `OrderFlowProvider` + implementacje.
  Kontrakt tradu: `{"ts": ms, "price": float, "qty": float, "side": +1|-1}`
  (+1 = agresywne kupno / -1 = agresywna sprzedaż).
- `build_flow_features.py` — trady → cechy per świeca 1h (`of_delta`, `of_cvd`,
  `of_delta_pct`, `of_big_delta` [footprint], `of_absorption`).
- `collect_orderflow.py` — kolektor inkrementalny (append-delta, dedup).

## Aktywne / gotowe źródła
| provider | stan | dane | koszt |
|---|---|---|---|
| binance | **AKTYWNY** | aggTrades trade-level (REST) | darmowy |
| mmt | gotowy (szkielet) | volume delta+buckety, footprint, liq/OB heatmapy | $199/mc, `MMT_API_KEY` |
| bybit/okx… | miejsce w rejestrze | — | — |

## Przełączenie źródła
```bash
ORDERFLOW_PROVIDER=binance   # domyślne (darmowe)
ORDERFLOW_PROVIDER=mmt MMT_API_KEY=xxx   # gdy wykupione
ORDERFLOW_SYMBOLS=BTC        # reżim BTC-only; rozszerzalne CSV
```
mmt: po wykupieniu uzupełnić endpoint/parametry w `MMTProvider.fetch_trades`
wg docs `/api` — kontrakt wyjścia (`ts/price/qty/side`) MUSI zostać, wtedy
`build_flow_features` i reszta działają bez zmian.

## Uruchomienie
```bash
python3 collect_orderflow.py --backfill 48   # seed 48h historii
python3 collect_orderflow.py                 # inkrement (cron co 1h)
```
Wynik: `orderflow/{provider}/{SYM}.parquet`.

## Następny krok (gen.Flow modele)
Cechy `of_*` → wpiąć do `ml_trainer.MODEL_FEATURES` (nowy zestaw „flow") +
etykiety REAKTYWNE przy HTF POI zamiast ślepego horyzontu → trening
sekwencyjny (TCN/Transformer) na podzie GPU. To atakuje sufit drzew (~66%)
od strony mikrostruktury.
