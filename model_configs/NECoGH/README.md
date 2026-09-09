# NECoGH — NewEdgeCampaign on GitHub — pakiety do AUDYTU (2026-08-05)

**STATUS: przygotowane do przeglądu, NIE URUCHOMIONE.** Żaden workflow GH nie został zdispatchowany
z tymi configami. Kod `ds_*`/`z_*` (warstwy 2/3) **jeszcze nie istnieje** w `features.py`/
`ml_trainer.py` — to tylko struktura configów gotowa do audytu, zanim ruszy implementacja.

## 3 paczki (kumulatywne, wg `raporty/newedge.md`)

**Paczka 1 — rptr (ZAIMPLEMENTOWANE, testowane):**
`CAT-sniper-rptr.json`, `LGB-sniper-rptr.json`, `XGB-sniper-rptr.json`, `RF-sniper-rptr.json`,
`ET-sniper-rptr.json` — rdzeń 11 + warstwa rptr (r_*/e_*). CAT i LGB **kończą tutaj na stałe**
(wg tabeli newedge.md nie dostają ds_*/z_*).

**Paczka 2 — rptr + ds_* (NIEZAIMPLEMENTOWANE):**
`XGB-sniper-necds.json` (+4 ds_* interakcje), `RF-sniper-necds.json` (+4 ds_* flow),
`ET-sniper-necds.json` (+23 ds_* pełny) — wymaga `calc_ds_features()`.

**Paczka 3 — rptr + ds_* + z_* (NIEZAIMPLEMENTOWANE):**
`RF-sniper-necz.json` (+4 z_*), `ET-sniper-necz.json` (+12 z_* pełny) — wymaga też
`calc_z_features()`. XGB/CAT/LGB nie mają wariantu paczki 3 (nie dostają z_* wg newedge.md).

## Ważne — co się stanie jeśli ktoś to uruchomi TERAZ

`feature_mix` w `hai_wfv.py` ma safe-filtering — brakujące kolumny (`ds_*`/`z_*`, bo kod nie
istnieje) zostaną po cichu odrzucone z warningiem `"feature-mix add ma brakujace kolumny [...] —
pominieto"`. Efekt: paczka 2/3 uruchomiona dziś da **dokładnie te same wyniki co paczka 1**
(realnie tylko rptr). Nie jest to błąd — to zamierzone bezpieczne zachowanie — ale nie testuje
tego co nazwa configu sugeruje, dopóki `calc_ds_features`/`calc_z_features` nie zostaną napisane.

## Docelowy dobór cech per model (finalna tabela, cel)

| Model | Paczka 1 | Paczka 2 (+ds_*) | Paczka 3 (+z_*) | Razem cel |
|---|---|---|---|---|
| CatBoost | 15 | — | — | 15 |
| LightGBM | 15 | — | — | 15 |
| XGBoost | 15 | +4 | — | ~19 |
| RandomForest | 24 | +4 | +4 | ~30 |
| ExtraTrees | 37 | +23 | +12 | ~72 |

Źródła: `raporty/newedge.md`, `raporty/RAPORT_NEWEDGE_CAMPAIGN.md`, `raporty/grok.txt`.
