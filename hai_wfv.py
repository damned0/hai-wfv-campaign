#!/usr/bin/env python3
"""HAI WFV — UCZCIWA walk-forward validation z retreningiem per okno.

    python3 hai_wfv.py --configs GC-div-h48,GC-rich-h48
    python3 hai_wfv.py --all
    python3 hai_wfv.py --configs X --windows 12 --window-days 45 --embargo 7
"""
from pathlib import Path
import os, sys
_HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HAI_ROOT", str(_HERE)))
INSTANCE_DIR = Path(os.environ.get("HAI_INSTANCE_DIR", str(ROOT / "HAI-NT" / "EPV")))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "hai_common"))
sys.path.insert(0, str(_HERE / "HAI-NL"))
sys.path.insert(0, str(INSTANCE_DIR))
import argparse, json, logging, os, sqlite3, time, uuid, subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# 
# # === DLACZEGO TEN PLIK POWSTAŁ (2026-07-13) ===
# `backtester.run_wfv()` NIE JEST walk-forward validation, mimo nazwy, parametru
# `embargo_days` i docstringa mówiącego o "oknie treningowym". Ładuje gotowe
# modele z model_registry RAZ (linia ~1585: `if not ensemble.active:
# ensemble.load_models()`) i NIGDY ich nie retrenuje. Skutek:
# 
#     modele trenowane 2026-07-09  ×  okna testowe od 2024-11-03
#     → model widział KAŻDE okno testowe w swoim zbiorze treningowym
# 
# To jest podręcznikowy LOOKAHEAD BIAS. Wszystkie PF (7.99 / 6.35 / 4.21) z tego
# harnessu są zawyżone, bo model zna przyszłość. Mimo tej przewagi min_pf
# wychodził 0.00 — czyli realnie modele są jeszcze gorsze.
# 
# Ten plik robi to, co run_wfv tylko UDAWAŁ:
#   dla każdego okna W:
#       cutoff   = start(W) - embargo_days
#       TRENUJ modele configu WYŁĄCZNIE na danych z timestamp < cutoff
#       symuluj trade'y na oknie W (out-of-sample — model tych danych nie widział)
# 
# `backtester.run_wfv` zostaje osobno jako in-sample backtest (hai_backtest.py),
# jawnie oznaczony — bo do rankingu względnego bywa użyteczny i jest 12× szybszy.
# 
# Wyniki → baza HAI-NL (hairesearch.db, wfv_runs) z harness='honest_v1' i
# lookahead_safe=1. Stare 679 wpisów (harness bez retreningu) skasowane 13.07.
# Sciezki parametryzowane env — ten sam kod dziala na VPS i na RunPodzie.
#   VPS:  HAI_ROOT=/root/ProjektHAI           HAI_INSTANCE_DIR=<root>/HAI-NT/EPV
#   pod:  HAI_ROOT=/workspace/ProjektHAIoRP   HAI_INSTANCE_DIR=<root>/WFV1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hai_wfv")

DB = Path(os.environ.get("HAI_DB", str(ROOT / "HAI-NL" / "data" / "hairesearch.db")))
CONFIGS_DIR = ROOT / "model_configs"
DATASET_CACHE = Path(os.environ.get(
    "HAI_WFV_CACHE", str(ROOT / "data_warehouse" / "meta" / "wfv_dataset.parquet")))

HARNESS = "honest_v1"

# Ilu symboli liczyc rownolegle w symulacji. Na VPS (6 rdzeni) zostaw 6;
# na RunPodzie (256 rdzeni, 4 shardy) 32 per shard = 128 watkow lacznie.
SIM_WORKERS = int(os.environ.get("HAI_SIM_WORKERS", "8"))

# ── Mapa: nazwa modelu z configu -> (rdzeń, horyzont[h], profil cech) ─────────
# Modele w registry były trenowane skryptami POZA kanonicznym trenerem
# (train_prec.py, train_farm.py, train_pod_v2.py), każdy z własną konwencją.
# 51 z 66 nazw używanych w configach NIE MA wpisu w ml_trainer.MODEL_FEATURES,
# więc trener sam z siebie ich nie odtworzy. Ta mapa przywraca semantykę:
# train_model(suffix, only=[core], horizon_hours=H, features=PROFIL) — dokładnie
# tak, jak je kiedyś wytrenowano (patrz train_prec.py:52-56).
CORES = ("cat", "xgb", "rf", "et", "lgb", "histgb")

# Specjalista struktury/reversji (2026-07-28): model (ET) na WASKIM zestawie
# nowych cech + minimalny kontekst. Testujemy hipoteze: cechy sd_prox/bars_cross
# psuly min_PF wrzucone do KAZDEGO modelu (rozlanie), ale jako osobny, regime-
# wazony specjalista moga dac avg-edge bez psucia najgorszego okna.
# Nazwa modelu w configu: '<core>_sdspec' (np. 'et_sdspec'). horyzont = 24h.
SD_SPEC_FEATURES = [
    "sd_prox", "bars_cross",          # rdzen (zweryfikowane importance 3-6/11)
    "atr_pct", "adx_14",              # rezim zmiennosci + sila trendu
    "rsi_4h", "price_position_bb",    # momentum + pozycja w kanale
]
# BATERIA KONTROLNA (recenzja Kimi3, 2026-07-28) — rozstrzyga czy lift specjalisty
# to CECHA czy ARTEFAKT ensemble.
#  - sdabl: core-only (BEZ sd_prox/bars_cross) -> izoluje "dodanie modelu" od "cechy"
#  - plac:  6 losowych ISTNIEJACYCH cech -> czysty efekt diversity/re-wazenia
SD_ABL_FEATURES = ["atr_pct", "adx_14", "rsi_4h", "price_position_bb"]
PLACEBO_FEATURES = ["volume_zscore", "macd_hist", "ema_fast_r",
                    "us10y_chg", "gold_chg", "hour_cos"]
# Profil SNIPER (cat_sniper/lgb_sniper itd., deploy 2026-08-01). Zestaw 11 cech
# z produkcji model_registry/gen.SNPR/6h/cat_sniper_6h.pkl, etykieta fast6h.
SNIPER_FEATURES = [
    "rsi_4h", "rsi_1d", "ema_mid_r", "atr_pct", "momentum",
    "trend_1d", "trend_4h", "adx_14", "volume_ratio",
    "price_position_bb", "bb_bandwidth_pct",
]
# Profil TREND/REV (cat_trend_6h/cat_rev_6h itd.) — zestawy z produkcji
# model_registry/gen.SNPR/6h/*.pkl (fakt z plikow, GRANDKAMPANIA §1, 2026-08-04).
# BUG naprawiony 2026-08-04: do tej pory train_window() uzywal SNIPER_FEATURES
# dla WSZYSTKICH trzech profili — "trend"/"rev" w honest WFV byly bez sensu
# duplikatem sniper pod inna nazwa (te same cechy, ta sama etykieta fast6h).
# 2026-08-11: WYCIETE `of_cvd_chg_24h` i `cvd_x_adx` — obie byly MARTWE, nie
# eksperymentalne. Zmierzone na cache 3.87 mln wierszy:
#   of_cvd_chg_24h — NIE MA go w datasecie (ml_trainer dokłada tę kolumnę tylko
#                    pod flagą HAI_LEJEK_CVD=1); w treningu i w live wchodziła
#                    jako 0.0 przez `features.get(f, 0.0)`
#   cvd_x_adx      — jest w cache, ale w 100% zer: ml_trainer.py:1120 ustawia
#                    `cvd_x_adx = 0.0` na sztywno, bo build_dataset nie wczytuje
#                    orderflow
# Model trendowy jechał więc realnie na 9 cechach z 11 — i tak było w treningu,
# w WFV i na produkcji. Cecha stała nie tworzy podziałów w drzewie, więc
# usunięcie jej nie zmienia predykcji; zmienia tylko to, że deklaracja zgadza
# się ze stanem faktycznym.
# Przywrócić dopiero gdy: (1) build_dataset zacznie czytać orderflow,
# (2) collect_orderflow obejmie więcej niż BTC (ORDERFLOW_SYMBOLS) i nadrobi
# zaległość — dane 128 symboli stoją na 2026-07-29.
TREND_FEATURES = [
    "trend_1h", "trend_4h", "trend_1d", "adx_14", "momentum", "rsi_4h",
    "volume_ratio", "taker_buy_ratio", "atr_pct",
]
REV_FEATURES = [
    "rsi", "rsi_4h", "ema_mid_r", "ema_slow_r", "price_position_bb",
    "bb_bandwidth_pct", "fib_dist_pct", "sr_dist_pct", "volume_zscore", "atr_pct",
]
PROFILE_FEATURES = {"sniper": SNIPER_FEATURES, "trend": TREND_FEATURES, "rev": REV_FEATURES}

# lgb_multi_horizon.pkl: horizons_trained = [4,6,8,16,24,36,48,72]
MULTI_HORIZONS = [4, 6, 8, 16, 24, 36, 48, 72]


def parse_model(name: str):
    """'cat_prec_h48' -> ('cat', 48, 'prec'). Zwraca None gdy nie umiemy odtworzyć."""
    core = next((c for c in sorted(CORES, key=len, reverse=True)
                 if name == c or name.startswith(c + "_")), None)
    if core is None:
        return None
    rest = name[len(core):].lstrip("_")

    profile = "base"
    for p in ("precB", "prec"):
        if rest.startswith(p):
            profile = p
            rest = rest[len(p):].lstrip("_")
            break

    # Profile deployowane 2026-08-01 (model_registry/gen.SNPR/6h): sniper/trend/rev.
    # Nazwy w configach: cat_sniper, lgb_trend, xgb_rev (male) oraz CAT_sniper_6h
    # (wielkie, configi 6h). Wszystkie = horyzont 6h, etykieta fast6h.
    _rest_lo = rest.lower().rstrip("_6h").lstrip("_")
    if _rest_lo in ("sniper", "trend", "rev"):
        return core, 6, _rest_lo

    # SPECJALISTA REZIM X KIERUNEK (2026-08-03): 'cat_sniper_r0L' / 'lgb_trend_r1S'.
    # modele trenowane TYLKO na wierszach danego rezymu ORAZ tylko nalongach (L) lub shortach (S).
    # Wzorzec: <core>_<sniper|trend|rev>_r<N>{L|S}
    import re as _re
    _m = _re.match(r"^(sniper|trend|rev)_r(\d)([lLsS])$", _rest_lo)
    if _m:
        _prof, _reg, _s = _m.group(1), _m.group(2), _m.group(3)
        return core, 6, f"regime{_reg}{_s.upper()}_{_prof}"

    horizon = 48  # domyślny (label_long, LOOKAHEAD_BARS=48)
    if rest.startswith("h") and rest[1:].split("_")[0].isdigit():
        horizon = int(rest[1:].split("_")[0])
    elif rest.startswith("fast") and rest[4:].rstrip("h").isdigit():
        horizon = int(rest[4:].rstrip("h"))
    elif rest.startswith("wide") and rest[4:].rstrip("h").isdigit():
        horizon = int(rest[4:].rstrip("h"))
    elif rest in ("sharp6x", ""):
        horizon = 48
    elif rest == "multi_horizon":
        # Model STACKED: jeden model uczony na 8 horyzontach naraz, z
        # horizon_hours jako CECHA wejsciowa (nie osobny model per horyzont).
        # f1=0.597 / precision=0.605 — NAJLEPSZY w calej puli (reszta 0.44-0.51).
        return core, MULTI_HORIZONS, "multi"
    elif rest.startswith("regime") and rest[6:].isdigit():
        # Specjalista rezimu rynku: trenowany TYLKO na wierszach danego rezimu
        # (regime_detector). Rezim liczony online z OHLCV.
        return core, 48, f"regime{rest[6:]}"
    elif rest == "sdspec":
        # Specjalista struktury/reversji na SD_SPEC_FEATURES, label 24h.
        return core, 24, "sdspec"
    elif rest == "sdabl":
        return core, 24, "sdabl"     # ablacja: core-only (bez sd/bars)
    elif rest == "plac":
        return core, 24, "plac"      # placebo: 6 losowych cech
    elif rest.startswith("sniper") or rest.startswith("trend") or rest.startswith("rev"):
        # WARIANTY kanoniczne 2026-08-04: <algo>_<sniper|trend|rev>_<H>[.<variant>]
        # np. et_sniper_v1, xgb_sniper_deploy, lgb_sniper_honest -> profil + 6h.
        _prof = "sniper" if rest.startswith("sniper") else ("trend" if rest.startswith("trend") else "rev")
        return core, 6, _prof
    elif rest.startswith("fit2"):
        # FIT2-div (B2 gen.Dir-v1): profil 'fit2', nieodtwarzalny (brak cech w trenerze),
        # ale nazwa spójna kanonicznie. Default 48h (label_long).
        return core, 48, "fit2"
    elif rest:
        return None  # np. et_h72_COREv2_iter2 — wariant nieodtwarzalny

    return core, horizon, profile


def load_config(name: str):
    p = CONFIGS_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("models", [])


def config_wagi_klas(name: str) -> "dict | None":
    """Wagi klas {0:NEUTRAL, 1:LONG, 2:SHORT} z configu, albo None (domyslne).

    Dodane 2026-08-27. Modele produkcyjne okazaly sie jednostronne: p99 dla
    P(long) to 0.33-0.39, ponizej bramki glosu 0.40 — live nie moze otworzyc
    pozycji na wzrost, 40 transakcji z rzedu bylo shortami. Etykiety maja tylko
    +7% przewagi SHORT, ale przy 93.6% klasy NEUTRAL ta drobna przewaga
    wzmacnia sie w rozkladzie wyjsciowym. Wagi pozwalaja to wyrownac przy
    treningu, zamiast obnizac progi (co wpuszcza takze slabe shorty).

    Klucze w JSON sa tekstem — zamieniamy na int, bo sklearn wymaga etykiet.
    """
    p = ROOT / "model_configs" / f"{name}.json"
    if not p.exists():
        return None
    w = json.loads(p.read_text()).get("wagi_klas")
    if not w:
        return None
    return {int(k): float(v) for k, v in w.items()}


def config_feature_mix(name: str) -> dict:
    """Zwraca feature_mix z configu (GRANDKAMPANIA §3.2 / THC §9): per-model modyfikacje cech.
    np. {"cat_sniper_6h": {"add": ["oi_change_24h"], "remove": ["bb_bandwidth_pct"]}}."""
    p = CONFIGS_DIR / f"{name}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("feature_mix") or {}
    except Exception:
        return {}


def _apply_symbol_whitelist(df):
    """Filtruje dataset do HAI_SYMBOLS (CSV) — lekki WFV A/B na podzbiorze coinow.
    Puste = caly dataset (stare zachowanie)."""
    wl = os.getenv("HAI_SYMBOLS", "").strip()
    if wl and "symbol" in df.columns:
        keep = {s.strip().upper() for s in wl.split(",") if s.strip()}
        n0 = len(df)
        df = df[df["symbol"].str.upper().isin(keep)].reset_index(drop=True)
        log.info(f"HAI_SYMBOLS: {df['symbol'].nunique()} coinow, {len(df):,}/{n0:,} wierszy")
    return df


# Skrypty, ktorych zmiana oznacza INNA SEMANTYKE cech w cache'u. Lista celowo
# wezsza niz w tools/waliduj_rejestr — tu chodzi wylacznie o to, co wchodzi do
# datasetu.
_ZRODLA_SEMANTYKI = [
    "hai_common/hai_common/ml_trainer.py",
    "data_warehouse/licz_korelacje_btc.py",
    "data_warehouse/licz_cechy_przekrojowe.py",
    "data_warehouse/licz_cechy_cvd.py",
]


def _cache_przeterminowany(cache: Path) -> list:
    """Czy ktorykolwiek skrypt liczacy cechy jest nowszy niz cache.

    POWOD (2026-08-24). build_dataset sprawdzal WYLACZNIE, czy cache ma
    potrzebne horyzonty etykiet. Po wpieciu cech CVD przebudowa cache'u
    zwrocila plik z 2026-08-21 i zalogowala "dataset z cache: 3,908,143
    wierszy" — czyli wygladala na sukces, a nowych cech w nim nie bylo.
    Wykryte przypadkiem, odruchowym uruchomieniem tools/waliduj_cache.py.

    Ten sam wzorzec trafil juz wczesniej: mapa cech policzona 2026-08-21 na
    cache'u sprzed naprawy przecieku dala oi_zscore_30d sile 0.883 zamiast
    0.242 — i ta cecha trafila jako kotwica do czterech modeli P4.

    Sprawdzenie kosztuje kilka stat(), wiec robimy je zawsze zamiast liczyc
    na to, ze ktos pamieta o osobnym walidatorze.

    Swieze SWIECE celowo NIE unieważniaja cache'u — cron dociaga je co
    godzine i cache zawsze bylby o kilka godzin starszy. Chodzi o zmiane
    ZNACZENIA cech, nie o ich zakres.
    """
    try:
        t_cache = cache.stat().st_mtime
    except OSError:
        return []
    mlodsze = []
    for wzgl in _ZRODLA_SEMANTYKI:
        p = ROOT / wzgl
        try:
            t = p.stat().st_mtime
        except OSError:
            continue
        if t > t_cache:
            mlodsze.append((f"skrypt {p.name}", datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")))

    # Katalogi z policzonymi cechami. Zapisuje do nich WYLACZNIE recznie
    # uruchamiany skrypt (licz_cechy_*), nigdy cron — wiec ich mtime oznacza
    # swiadome przeliczenie, a nie dociagniecie swiec. Bez tego cache przezyl
    # 2026-08-26 przebudowe calej historii order flow: skrypty sie nie zmienily,
    # wiec bramka oparta na samych plikach .py niczego nie zauwazyla.
    for kat in ("data_warehouse/cechy_cvd", "data_warehouse/cechy_przekrojowe",
                "data_warehouse/korelacje_btc"):
        d = ROOT / kat
        if not d.exists():
            continue
        try:
            t = max(p.stat().st_mtime for p in d.glob("*.parquet"))
        except (OSError, ValueError):
            continue
        if t > t_cache:
            mlodsze.append((f"cechy {d.name}", datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")))
    return mlodsze


def build_dataset(horizons):
    """Dataset z etykietami dla WSZYSTKICH potrzebnych horyzontów naraz.
    Budowany RAZ (drogie: ~106 symboli × cechy), cache'owany na dysk."""
    from core import ml_trainer as mt

    if DATASET_CACHE.exists():
        df = pd.read_parquet(DATASET_CACHE)
        _LBL2H = {"label_fast6h": 6, "label_fast24h": 24, "label_long": 48,
                  "label_h72": 72, "label_wide96h": 96}
        have = {h for c, h in _LBL2H.items() if c in df.columns}
        # etykiety extra_horizons maja postac label_h{N} (np. label_h4, label_h16)
        for c in df.columns:
            if c.startswith("label_h") and c[7:].isdigit():
                have.add(int(c[7:]))
        if set(horizons) <= have:
            przeterm = _cache_przeterminowany(DATASET_CACHE)
            if przeterm:
                log.warning("CACHE NIEAKTUALNY — %s nowszych niz cache; przebudowa", len(przeterm))
                for opis, kiedy in przeterm:
                    log.warning("    %s  %s", kiedy, opis)
            else:
                df = _apply_symbol_whitelist(df)
                log.info(f"dataset z cache: {len(df):,} wierszy, horyzonty {sorted(have)}")
                return df
        else:
            log.info(f"cache ma horyzonty {sorted(have)}, potrzebne {sorted(horizons)} — przebudowa")
        log.info(f"cache ma horyzonty {sorted(have)}, potrzebne {sorted(horizons)} — przebudowa")

    # Standardowe etykiety (6/24/48/72/96) build_features liczy zawsze.
    # Reszta (4/8/16/36 — dla lgb_multi_horizon) idzie jako extra_horizons.
    std = {6, 24, 48, 72, 96}
    extra = sorted(h for h in horizons if h not in std)
    log.info(f"buduję dataset (horyzonty {sorted(horizons)}"
             f"{f', extra: {extra}' if extra else ''}) — to potrwa…")
    t0 = time.time()
    df = mt.build_dataset(extra_horizons=extra) if extra else mt.build_dataset()
    # extra_horizons wchodzi przez build_features_for_symbol; gdy build_dataset
    # go nie przyjmuje, etykiety dla innych H liczymy z triple_barrier poniżej.
    DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_CACHE, index=False)
    log.info(f"dataset: {len(df):,} wierszy w {time.time()-t0:.0f}s → {DATASET_CACHE}")
    return _apply_symbol_whitelist(df)


def add_regime_column(df):
    """Dokłada kolumnę `regime` (stan HMM per świeca) — potrzebna do odtworzenia
    et_regime1/2 i rf_regime1/2 (specjaliści reżimu, trenowani TYLKO na wierszach
    danego reżimu). Dataset jej nie miał, więc te 4 modele były nieodtwarzalne.

    RegimeDetector._build_features() liczy cechy WEKTOROWO dla całej serii, a HMM
    ma .predict() na sekwencji — więc reżim da się policzyć dla każdej świecy,
    nie tylko dla ostatniej (predict_online zwraca tylko bieżący stan)."""
    if "regime" in df.columns:
        return df
    try:
        from core.regime_detector import RegimeDetector
        import joblib
        rd = RegimeDetector()
        pkl = INSTANCE_DIR / "data" / "models" / "regime_hmm.pkl"
        if pkl.exists():
            d = joblib.load(pkl)
            rd.model = d.get("model") or d
            rd.scaler = d.get("scaler")
            rd.is_trained = True
    except Exception as e:
        log.warning(f"regime: nie udało się wczytać HMM ({e}) — modele *_regime* zostaną pominięte")
        return df
    if not getattr(rd, "is_trained", False) or rd.model is None:
        log.warning("regime: brak wytrenowanego HMM — modele *_regime* zostaną pominięte")
        return df

    from core.backtester import backtester
    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        c1h = backtester._load_window(sym, "1h", 3650, 0)
        if not c1h:
            continue
        o = pd.DataFrame(c1h)
        try:
            feats = rd._build_features(o)
            if feats is None:
                continue
            states = rd.model.predict(rd._normalize_features(feats))
            # _load_window zwraca timestamp jako int64 w MILISEKUNDACH.
            # Bez unit="ms" pandas czyta to jako nanosekundy -> 1970-01-01,
            # merge nie trafia w zaden wiersz, wszystko dostaje regime=-1,
            # a modele *_regime* padaja na pustym zbiorze (bug 2026-07-14).
            _ts = o["timestamp"]
            _ts = pd.to_datetime(_ts, unit="ms") if pd.api.types.is_integer_dtype(_ts) \
                  else pd.to_datetime(_ts)
            if getattr(_ts.dt, "tz", None) is not None:
                _ts = _ts.dt.tz_localize(None)
            m = pd.DataFrame({"timestamp": _ts, "regime": states})
            gg = g.merge(m, on="timestamp", how="left")
            parts.append(gg)
        except Exception:
            continue
    if not parts:
        log.warning("regime: nie policzono — modele *_regime* zostaną pominięte")
        return df
    out = pd.concat(parts, ignore_index=True)
    out["regime"] = out["regime"].fillna(-1).astype(int)
    log.info(f"regime: policzony dla {len(out):,} wierszy | rozkład: "
             f"{out['regime'].value_counts().to_dict()}")
    return out


def label_col(horizon: int) -> str:
    std = {6: "label_fast6h", 24: "label_fast24h", 48: "label_long",
           72: "label_h72", 96: "label_wide96h"}
    return std.get(horizon) or f"label_h{horizon}"   # extra_horizons -> label_h{H}


def train_window(df_train, models, mt, mix=None, wagi_klas=None):
    """Trenuje modele configu WYŁĄCZNIE na df_train (dane sprzed cutoffu).
    mix: {model_name: {"add": [...], "remove": [...]}} — feature-mix per model
    (GRANDKAMPANIA §3.2, THC §9): nakłada modyfikacje na domyślne cechy profilu.

    wagi_klas (2026-08-27): {0:..,1:..,2:..} — waga NEUTRAL/LONG/SHORT przy
    treningu. Powod: modele produkcyjne okazaly sie jednostronne — p99 dla
    P(long) wynosi 0.33-0.39, czyli PONIZEJ bramki glosu 0.40, wiec live nie
    moze otworzyc pozycji na wzrost. Zmierzone: 40 transakcji, 100% shortow.
    Domyslne {0:1.0, 1:2.5, 2:2.5} traktuja oba kierunki rowno, ale etykiety
    maja +7% przewagi SHORT, a rzadka klasa wzmacnia te przewage w rozkladzie
    wyjsciowym. Podnoszac wage LONG mozna to wyrownac BEZ ruszania progow."""
    mix = mix or {}
    trained = {}
    for name in models:
        spec = parse_model(name)
        if spec is None:
            log.warning(f"    {name}: nie umiem odtworzyć — POMIJAM (config niepełny!)")
            continue
        core, horizon, profile = spec

        # STACKED multi-horizon: sklej kopie datasetu (po jednej na horyzont),
        # etykieta = label tego horyzontu, horizon_hours = CECHA.
        if profile == "multi":
            frames = []
            # RAM-safe (2026-08-08): df_train ~3.87M×98 (~3GB) — kopiowanie go
            # 8× = ~24GB -> OOM na VPS 11GB. Okrawamy do kolumn CECH modelu
            # (+label), subsamplujemy PER HORYZONT przed concat (suma <= cap).
            _cap = int(os.environ.get("HAI_MULTI_MAX_ROWS", "4000000"))
            _per = max(_cap // max(len(horizon), 1), 50000)
            _madd = set((mix.get(name) or {}).get("add", []))
            _mrm = set((mix.get(name) or {}).get("remove", []))
            _want = (set(mt.MODEL_FEATURES.get(core) or []) | _madd | {"timestamp"}) - _mrm
            for h in horizon:
                lc = label_col(h)
                if lc not in df_train.columns:
                    continue
                part = df_train[[c for c in _want if c in df_train.columns] + [lc]].copy()
                part["horizon_hours"] = float(h)
                part["label_stacked"] = part[lc]
                if len(part) > _per:
                    part = part.sample(n=_per, random_state=42)
                frames.append(part)
            if not frames:
                log.warning(f"    {name}: brak etykiet dla {horizon} — POMIJAM")
                continue
            df_use = pd.concat(frames, ignore_index=True).sort_values("timestamp")
            log.info(f"    {name}: stacked {len(frames)}x = {len(df_use):,} wierszy "
                     f"(cap {_cap:,}, per-horyzont {_per:,})")
            lbl = "label_stacked"
            feats = (mt.MODEL_FEATURES.get(core) or []) + ["horizon_hours"]
        elif profile.startswith("regime"):
            # Format 1 (istniejacy): regime<N> -> tylko rezim
            # Format 2 (2026-08-03, rezim x kierunek): regime<N><L|S>_<sniper|trend|rev>
            #   -> trening tylko na wierszach rezimu N, i wyłącznie na LONG (L) / SHORT (S).
            if "regime" not in df_train.columns:
                log.warning(f"    {name}: dataset bez kolumny 'regime' — POMIJAM "
                            f"(regime specialists wymagaja przebudowy datasetu)")
                continue
            import re as _re2
            _rm = _re2.match(r"^regime(\d)([LS])_(sniper|trend|rev)$", profile)
            if _rm:
                want = int(_rm.group(1))
                _side = _rm.group(2)
                df_use = df_train[df_train["regime"] == want]
                if len(df_use) < 1000:
                    log.warning(f"    {name}: za malo probek dla regime={want} ({len(df_use)}) — POMIJAM")
                    continue
                lbl = label_col(horizon)   # e.g. 6h -> label_fast6h
                feats = SNIPER_FEATURES
                # kierunek: zostaw tylko LONG (1) lub SHORT (2), neutral (0) wyrzuc
                _want_cls = 1 if _side == "L" else 2
                if lbl not in df_use.columns:
                    log.warning(f"    {name}: brak kolumny {lbl} — POMIJAM")
                    continue
                df_use = df_use[df_use[lbl] == _want_cls]
                if len(df_use) < 300:
                    log.warning(f"    {name}: za malo probek dla {lbl}=={_want_cls} ({len(df_use)}) — POMIJAM")
                    continue
                log.info(f"    {name}: regim={want} kierunek={'LONG' if _side=='L' else 'SHORT'} "
                         f"({len(df_use)} probek)")
            else:
                # jak dotad: czysty regime<N>
                want = int(profile[6:])
                df_use = df_train[df_train["regime"] == want]
                if len(df_use) < 5000:
                    log.warning(f"    {name}: za malo probek dla regime={want} ({len(df_use)}) — POMIJAM")
                    continue
                lbl, feats = label_col(horizon), None
        elif profile == "sdspec":
            df_use = df_train
            lbl = label_col(horizon)          # 24h
            feats = SD_SPEC_FEATURES          # waski zestaw specjalisty
        elif profile in ("sniper", "trend", "rev"):
            df_use = df_train
            lbl = label_col(6)                # fast6h
            feats_wanted = PROFILE_FEATURES[profile]
            feats = [f for f in feats_wanted if f in df_train.columns]
            _missing = [f for f in feats_wanted if f not in df_train.columns]
            if _missing:
                log.warning(f"    {name}: brak kolumn {_missing} w datasecie — "
                            f"{profile} jedzie na {len(feats)}/{len(feats_wanted)} cech")
        elif profile == "sdabl":
            df_use = df_train; lbl = label_col(horizon); feats = SD_ABL_FEATURES
        elif profile == "plac":
            df_use = df_train; lbl = label_col(horizon); feats = PLACEBO_FEATURES
        else:
            df_use = df_train
            lbl = label_col(horizon)
            feats = mt.PREC_FEATURES if (profile in ("prec", "precB")
                                          and hasattr(mt, "PREC_FEATURES")) else None

        # FEATURE-MIX (GRANDKAMPANIA §3.2 / THC §9): nakłada modyfikacje z configu
        # na domyślne cechy profilu. np. {"add":["oi_change_24h"], "remove":["bb_bandwidth_pct"]}
        _mix = mix.get(name)
        if _mix:
            _fl = list(feats) if feats else []
            for _add in _mix.get("add", []):
                if _add not in _fl:
                    _fl.append(_add)
            _rm = set(_mix.get("remove", []))
            _fl = [f for f in _fl if f not in _rm]
            # Bezpieczny filtr (2026-08-04, ta sama zasada co PROFILE_FEATURES
            # nizej) - "add" moze wskazywac kolumny ktorych nie ma w tym
            # datasecie (np. cvd_x_adx/of_cvd_chg_24h) - filtrujemy zamiast
            # crashowac na selekcji kolumn.
            _fl_ok = [f for f in _fl if f in df_use.columns]
            _fl_missing = [f for f in _fl if f not in df_use.columns]
            # FIX 2026-08-07: bylo "warning + jedziemy dalej". Kampania RPGC v8
            # przez 20h liczyla NIE TO, CO TRZEBA: pod_build_cache.py pisal cache
            # do wfv_dataset_rptr.parquet, a build_dataset czytal wfv_dataset.parquet
            # (HAI_WFV_CACHE nikt nie ustawial), wiec dataset NIE MIAL kolumn
            # r_*/e_*/x_*/funding_*. Filtr grzecznie je wyrzucil, modele pojechaly
            # na 7-17 starych cechach, ET zostal z garstka i dal ZERO transakcji.
            # Wszystko wygladalo zdrowo do samego konca.
            # feature_mix to JAWNA deklaracja intencji - brak kolumny to blad
            # konfiguracji/danych, nie stan do zignorowania. Ginemy glosno.
            if _fl_missing:
                log.error(f"    {name}: feature-mix zadeklarowal {len(_fl)} cech, "
                          f"dataset ma {len(_fl_ok)}. BRAKUJE: {_fl_missing}")
                if os.environ.get("HAI_ALLOW_PARTIAL_MIX") != "1":
                    raise SystemExit(
                        f"PRZERWANO ({name}): dataset nie ma kolumn {_fl_missing}. "
                        f"Przebuduj cache (scripts/pod_build_cache.py) albo — jesli "
                        f"naprawde chcesz liczyc na okrojonym zestawie — ustaw "
                        f"HAI_ALLOW_PARTIAL_MIX=1.")
                log.warning(f"    {name}: HAI_ALLOW_PARTIAL_MIX=1 — jade na "
                            f"{len(_fl_ok)}/{len(_fl)} cechach ŚWIADOMIE")
            feats = _fl_ok
            log.info(f"    {name}: feature-mix -> {len(feats)} cech: {feats}")

        if lbl not in df_use.columns:
            log.warning(f"    {name}: brak kolumny {lbl} w dataset — POMIJAM")
            continue

        _orig_lbl = mt.MODEL_LABEL_COLUMN.get(core)
        _orig_feat = mt.MODEL_FEATURES.get(core)
        try:
            mt.MODEL_LABEL_COLUMN[core] = lbl
            if feats:
                mt.MODEL_FEATURES[core] = feats
            res = mt.train_models(df_use, only=[core], class_weights=wagi_klas)
            if core in res and "model" in res[core]:
                trained[name] = res[core]
        except Exception as e:
            log.warning(f"    {name}: trening padł ({e})")
        finally:
            if _orig_lbl is None:
                mt.MODEL_LABEL_COLUMN.pop(core, None)
            else:
                mt.MODEL_LABEL_COLUMN[core] = _orig_lbl
            if _orig_feat is not None:
                mt.MODEL_FEATURES[core] = _orig_feat
    return trained


def inject(ensemble, trained):
    """Wstrzykuje świeżo wytrenowane modele do ensemble (bez tykania registry).
    Format train_models() == format, który load_models() czyta z .pkl."""
    ensemble.models.clear(); ensemble.scalers.clear()
    ensemble.feature_names.clear(); ensemble.f1_scores.clear()
    ensemble.precision_scores.clear(); ensemble.accuracies.clear()
    for name, d in trained.items():
        ensemble.models[name] = d["model"]
        ensemble.scalers[name] = d.get("scaler")
        ensemble.feature_names[name] = d.get("features")
        ensemble.f1_scores[name] = d.get("f1", 0.0)
        ensemble.precision_scores[name] = d.get("precision", 0.0)
        ensemble.accuracies[name] = d.get("accuracy", 0.0)
    ensemble._recalc_weights()
    ensemble.active = bool(ensemble.models)


_SCHEMA_WINDOWS = """create table if not exists wfv_windows(
  id integer primary key autoincrement, run_id text, model_config text, window text,
  total_trades int, wins int, losses int, win_rate real, total_pnl_usdt real,
  profit_factor real, max_drawdown_pct real, avg_hold_hours real, longs int, shorts int,
  sharpe_ratio real, syms_pf_gt15_pct real, syms_tested int, circuit_breaker_days int,
  train_cutoff text, saved_at text)"""

_SCHEMA_TRADES = """create table if not exists wfv_trade_log(
  id integer primary key autoincrement,
  run_id text, model_config text, window text,
  symbol text, side text,
  entry real, exit_price real, pnl_net real, pnl_usdt real, pnl_pct real,
  result text, open_ts int, close_ts int, atr real, size_usdt real, hours_held real,
  had_pyramid int, pyramid_pnl real, bb_pos real, regime text, session text,
  confidence real, dominant_model text, model_votes text, feature_snapshot text)"""

_SCHEMA = """create table if not exists wfv_runs(
  id integer primary key autoincrement, source_file text, instance text, saved_at text,
  n_windows int, window_days int, mode text, regime_adaptive int, voting_mode text,
  decision_threshold real, threshold_long real, threshold_short real, conf_sizing int,
  meta_label int, consensus_min int, model_config text, models text, avg_pf real,
  min_pf real, max_dd real, avg_trades real, decision text, longs int, shorts int,
  ingested_at text, harness text, lookahead_safe int, vote_gate real, avg_wr real,
  sharpe real, weak_windows int, run_id text, train_cutoffs text,
  min_pf_holdout real, n_holdout int, conf text)"""


def _conn(retries: int = 6):
    """WAL + busy_timeout: 4 shardy pisza do JEDNEJ bazy rownolegle i domyslny
    SQLite (rollback journal, timeout 5s) wywalal `database is locked` —
    shardy 0 i 3 padly na tym 13.07. WAL pozwala czytac w trakcie zapisu,
    busy_timeout kaze czekac zamiast rzucac wyjatkiem, a retry lapie reszte."""
    DB.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for i in range(retries):
        try:
            c = sqlite3.connect(DB, isolation_level=None, timeout=60)
            c.execute("pragma journal_mode=WAL")
            c.execute("pragma busy_timeout=60000")
            c.execute("pragma synchronous=NORMAL")
            c.execute(_SCHEMA); c.execute(_SCHEMA_WINDOWS); c.execute(_SCHEMA_TRADES)
            # Migracja idempotentna (2026-07-29): kolumny holdout dla starych baz
            for _col, _typ in (("min_pf_holdout", "real"), ("n_holdout", "int")):
                try:
                    c.execute(f"alter table wfv_runs add column {_col} {_typ}")
                except sqlite3.OperationalError:
                    pass  # kolumna juz istnieje
            # Migracja 2026-08-03: kolumna 'conf' (prog min_confidence) dla retencji
            try:
                c.execute("alter table wfv_runs add column conf text")
            except sqlite3.OperationalError:
                pass
            # Migracja 2026-08-03: wfv_trade_log — dopisz run_id/model_config (starsze bazy
            # mialy wfvasnie schema bez tych kolumn, przez co save_trades failowal na
            # 'no such column: model_config' i tradesy NIE trafialy do bazy (brak danych do
            # dobierania cech). Dopisanie idempotentne dla nowych i istniejacych baz.
            for _tcol in ("run_id", "model_config"):
                try:
                    c.execute(f"alter table wfv_trade_log add column {_tcol} text")
                except sqlite3.OperationalError:
                    pass
            return c
        except sqlite3.OperationalError as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


def save_window(run_id, cfg, st, cutoff):
    """Metryki per OKNO. Wczesniej liczone i WYRZUCANE — do bazy szedl tylko
    zagregowany werdykt (8 liczb na config), mimo ze kazde okno daje komplet
    statystyk (sharpe, pnl, hold, longs/shorts, syms_pf...)."""
    c = _conn()
    cols = [r[1] for r in c.execute("pragma table_info(wfv_windows)")]
    rec = {"run_id": run_id, "model_config": cfg, "train_cutoff": cutoff,
           "saved_at": datetime.now().isoformat(), **{k: v for k, v in st.items()
                                                       if k in cols and k != "id"}}
    rec = {k: v for k, v in rec.items() if k in cols}
    try:
        c.execute(f"insert into wfv_windows ({','.join(rec)}) values ({','.join('?'*len(rec))})",
                  list(rec.values()))
    except sqlite3.OperationalError as e:
        log.warning(f"wfv_windows: zapis nieudany ({e}) — wynik jest w JSON sharda")
    finally:
        c.close()


def save_trades(run_id, cfg, window, trades):
    """KAZDY trade. Tabela wfv_trade_log istniala, ale NIKT jej nie wypelnial —
    znikaly wszystkie pojedyncze transakcje (a to z nich liczy sie atrybucja
    modeli/cech i analiza per symbol)."""
    if not trades:
        return
    c = _conn()
    cols = [r[1] for r in c.execute("pragma table_info(wfv_trade_log)")]
    # Mapowanie kluczy trade'a z backtestera -> kolumny realnej bazy (2026-08-03).
    # backtester zwraca: entry/exit/hours_held/result/open_ts/close_ts + dominant_model/model_votes
    _KEYS = ["run_id", "model_config", "window", "symbol", "side",
             "entry", "exit_price", "pnl_net", "pnl_usdt", "pnl_pct", "result",
             "open_ts", "close_ts", "atr", "size_usdt", "hours_held",
             "had_pyramid", "pyramid_pnl", "bb_pos", "regime", "session",
             "confidence", "dominant_model", "model_votes", "feature_snapshot"]
    rows = []
    for t in trades:
        rec = {
            "run_id": run_id, "model_config": cfg, "window": window,
            "symbol": t.get("symbol"), "side": t.get("side"),
            "entry": t.get("entry"), "exit_price": t.get("exit"),
            "pnl_net": t.get("pnl_net"), "pnl_usdt": t.get("pnl_usdt"), "pnl_pct": t.get("pnl_pct"),
            "result": t.get("result"), "open_ts": t.get("open_ts"), "close_ts": t.get("close_ts"),
            "atr": t.get("atr"), "size_usdt": t.get("size_usdt"), "hours_held": t.get("hours_held"),
            "had_pyramid": t.get("had_pyramid"), "pyramid_pnl": t.get("pyramid_pnl"),
            "bb_pos": t.get("bb_pos"), "regime": t.get("regime"), "session": t.get("session"),
            "confidence": t.get("confidence"),
            "dominant_model": t.get("dominant_model"),
            "model_votes": json.dumps(t.get("model_votes")) if isinstance(t.get("model_votes"), (dict, list)) else t.get("model_votes"),
            "feature_snapshot": json.dumps(t.get("feature_snapshot")) if isinstance(t.get("feature_snapshot"), (dict, list)) else t.get("feature_snapshot"),
        }
        rows.append([rec.get(k) for k in _KEYS if k in cols])
    keys = [k for k in _KEYS if k in cols]
    try:
        c.executemany(f"insert into wfv_trade_log ({','.join(keys)}) "
                      f"values ({','.join('?'*len(keys))})", rows)
    except Exception as e:
        # 2026-08-03: byl tu tylko sqlite3.OperationalError, ale realny blad
        # (dict jako parametr, niezserializowany model_votes/feature_snapshot)
        # to sqlite3.ProgrammingError - nie byl lapany, wywalal cala kampanie
        # (save_run() dla werdyktu nigdy nie byl osiagany po tym punkcie).
        log.warning(f"wfv_trade_log: zapis nieudany ({e})")
    finally:
        c.close()


def save_run(rec: dict):
    c = _conn()
    cols = [r[1] for r in c.execute("pragma table_info(wfv_runs)")]
    rec = {k: v for k, v in rec.items() if k in cols}
    c.execute(f"insert into wfv_runs ({','.join(rec)}) values ({','.join('?'*len(rec))})",
              list(rec.values()))
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--window-days", type=int, default=45)
    ap.add_argument("--embargo", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.35)
    # KOTWICA CZASU (2026-08-19): granice okien liczyly sie od datetime.now()
    # w chwili URUCHOMIENIA, wiec dwa przebiegi z roznych dni mialy INNE okna.
    # Zmierzone: baza (17.08 14:16) vs pomiar knotowy (18.08 10:31) = ~20h
    # przesuniecia; przy ~5 stratach na 314 transakcji wystarczy to, by zmienic
    # PF o dziesiatki. Bez kotwicy zaden A/B nie jest czysty — dotyczy calej
    # bazy 99 runow. Domyslnie None = stare zachowanie (wsteczna zgodnosc).
    ap.add_argument("--anchor", default=None, metavar="RRRR-MM-DD",
                    help="kotwica czasu dla granic okien (domyslnie: teraz)")
    # SWEEP PROGOW (2026-08-19): progi dzialaja przy DECYZJI, nie przy treningu
    # (bt_mod._DECISION_THRESHOLD ustawiane raz, globalnie) — wiec modele sa
    # IDENTYCZNE dla wszystkich progow. Sweep N progow jako N osobnych przebiegow
    # to N-krotne powtarzanie tego samego treningu. Zmierzone na podzie: trening
    # okna 816s, symulacja ~80s. Tutaj: trenuj RAZ, symuluj N razy.
    ap.add_argument("--threshold-sweep", default=None, metavar="0.20,0.30,...",
                    help="lista progow decyzji; wyniki jako <config>@t<prog>")
    ap.add_argument("--vote-gate", type=float, default=0.40)
    ap.add_argument("--mode", default="neutral")
    ap.add_argument("--conf", type=float, default=None,
                    help="Prog min_confidence (override). Gdy None -> STRATEGY_MIN_CONFIDENCE env "
                         "(jesli ustawione) -> domyslne z trybu. Konfigurowalne recznie lub z env.")
    # Holdout firewall (2026-07-29): rezerwuj N NAJNOWSZYCH okien jako holdout — nie
    # wchodza do decyzji GO/NO-GO, tylko finalna walidacja wybranego kandydata. Chroni
    # przed multiple-comparison bias: testujac wiele cech na tych samych oknach ryzykujesz
    # ze "ocalala" cecha to artefakt. Jesli przechodzi dev ale peka na holdout -> do kosza.
    ap.add_argument("--holdout", type=int, default=0,
                    help="N najnowszych okien jako holdout (poza GO/NO-GO)")
    # Okna sa od siebie CALKOWICIE niezalezne (kazde ma wlasny cutoff i wlasny
    # trening) — mozna je liczyc rownolegle. Na RunPodzie (256 rdzeni) jeden
    # proces zjadal ~13% CPU i 0% GPU; --shard i/N rozbija kampanie na N
    # procesow, kazdy bierze co N-te okno. Wyniki czastkowe z kazdego sharda
    # sa scalane po run_id (kazdy shard pisze swoje okna do tej samej bazy).
    ap.add_argument("--shard", default=None, help="i/N — licz tylko co N-te okno (np. 0/4)")
    ap.add_argument("--run-id", default=None, help="wspolny run_id dla shardow")
    args = ap.parse_args()

    if args.all:
        names = sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))
    elif args.configs:
        names = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        ap.error("podaj --configs albo --all")

    # Konfigurowalny prog confidence (2026-08-03): --conf (reczny) ma priorytet
    # nad STRATEGY_MIN_CONFIDENCE env, a ten nad domyslnym trybu.
    if args.conf is not None:
        os.environ["STRATEGY_MIN_CONFIDENCE"] = str(args.conf)
        log.info(f"STRATEGY_MIN_CONFIDENCE = {args.conf} (reczny --conf)")

    from core import ml_trainer as mt
    from core.backtester import backtester
    from core import backtester as bt_mod
    from core.ensemble import ensemble

    bt_mod._DECISION_THRESHOLD = args.threshold
    bt_mod._VOTE_GATE = args.vote_gate

    _SWEEP = None
    if args.threshold_sweep:
        _SWEEP = [float(x) for x in args.threshold_sweep.split(",") if x.strip()]
        log.info(f"SWEEP PROGOW: {_SWEEP} — trening RAZ na okno, symulacja x{len(_SWEEP)}")
    bt_mod._DOCTRINE_FREE = True

    # jakie horyzonty w ogóle będą potrzebne
    horizons = set()
    plan = {}
    for n in names:
        models = load_config(n)
        if not models:
            log.warning(f"{n}: brak configu — pomijam")
            continue
        specs = [(m, parse_model(m)) for m in models]
        bad = [m for m, s in specs if s is None]
        if bad:
            log.warning(f"{n}: nieodtwarzalne modele {bad} — config będzie NIEPEŁNY")
        _mix = config_feature_mix(n)
        _wagi = config_wagi_klas(n)
        if _wagi:
            log.info(f"{n}: wagi klas z configu {_wagi} (0=NEUTRAL 1=LONG 2=SHORT)")
        plan[n] = (models, _mix)
        if _mix:
            log.info(f"{n}: feature_mix aktywny ({len(_mix)} modeli z modyfikacjami)")
        for _, sp in specs:
            if not sp:
                continue
            h = sp[1]
            horizons |= set(h) if isinstance(h, list) else {h}

    if not plan:
        log.error("nic do policzenia")
        sys.exit(1)

    df = build_dataset(horizons)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    need_regime = any(parse_model(m) and str(parse_model(m)[2]).startswith("regime")
                      for ms, _mx in plan.values() for m in ms)
    if need_regime:
        df = add_regime_column(df)
    if args.anchor:
        now = datetime.strptime(args.anchor, "%Y-%m-%d")
        log.info(f"KOTWICA CZASU: {now:%Y-%m-%d} (okna liczone od tej daty, nie od teraz)")
    else:
        now = datetime.now(timezone.utc).replace(tzinfo=None)

    run_id = args.run_id or (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4])
    shard_i, shard_n = (0, 1)
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    log.info(f"=== HAI WFV (honest, retrening per okno) | run {run_id} | "
             f"{len(plan)} configów | {args.windows}×{args.window_days}d "
             f"embargo={args.embargo} thresh={args.threshold} ===")

    # Modele powtarzaja sie miedzy configami (np. cat_h48 jest w kilkunastu).
    # Trenowanie ich per config = ta sama robota N razy: 76 configow x 12 okien
    # x ~6 modeli = ~5500 trenowan zamiast ~730 unikalnych. Dlatego pętla jest
    # ODWROCONA: zewnetrzna = OKNO, wewnatrz trenujemy kazdy unikalny model RAZ,
    # a configi tylko skladaja gotowe modele i symuluja.
    per_window_results = {name: [] for name in plan}
    per_window_cutoffs = []

    for w in range(args.windows):
        if w % shard_n != shard_i:
            continue
        idx = args.windows - 1 - w
        off_end = idx * (args.window_days + args.embargo)
        off_start = off_end + args.window_days
        win_start = now - timedelta(days=off_start)
        cutoff = win_start - timedelta(days=args.embargo)

        df_train = df[df["timestamp"] < cutoff]
        if len(df_train) < 5000:
            log.warning(f"W{w+1}: za malo danych treningowych ({len(df_train)}) — pomijam okno")
            continue

        # FIX 2026-08-07: bylo bez warunku `not (_mx and m in _mx)` — model uzyty
        # w configu z feature_mix byl trenowany DWA RAZY: raz "czysto" do banku
        # (nikt tego nie uzywal, bo config siega po wersje mixowana) i raz z mixem.
        # Przy kampanii gdzie WSZYSTKIE configi maja feature_mix to dokladnie
        # 2x koszt treningu. Bierzemy do czystego banku tylko te modele, ktore
        # jakis config faktycznie chce bez mixu.
        needed = sorted({m for models, _mx in plan.values() for m in models
                         if parse_model(m) and not (_mx and m in _mx)})
        # Konfigi z feature_mix: modele w nich trenowane OSOBNO (zmienione cechy),
        # wiec dodajemy je pod kluczem (cfg, model) — nie koliduja z czystym bankiem.
        mix_models = {}
        for _cfg, (_models, _mx) in plan.items():
            if not _mx:
                continue
            for _m in _models:
                if _m in _mx:
                    mix_models.setdefault(_cfg, []).append(_m)
        log.info(f"\n=== OKNO W{w+1}/{args.windows} | trening < {cutoff:%Y-%m-%d} "
                 f"({len(df_train):,} probek) | test {win_start:%Y-%m-%d}..{(now-timedelta(days=off_end)):%Y-%m-%d} "
                 f"| {len(needed)} unikalnych modeli + {len(mix_models)} mixowanych ===")
        t_tr = time.time()
        bank = train_window(df_train, needed, mt)
        mix_bank = {}
        for _cfg, _mmodels in mix_models.items():
            _mmix = {m: plan[_cfg][1][m] for m in _mmodels}
            _mwagi = config_wagi_klas(_cfg)
            # FIX 2026-08-07: bylo mix_bank.update(...) — klucz = SAMA nazwa
            # modelu, wbrew komentarzowi wyzej ("pod kluczem (cfg, model)").
            # Dopoki kazdy config mial inny model (cat_/rf_/lgb_) nic sie nie
            # dzialo, ale przy dwoch configach mixujacych TEN SAM model drugi
            # po cichu nadpisywal pierwszy i oba liczyly sie na cechach tego
            # drugiego. Blokowalo to laczenie configow w jeden proces.
            for _m, _d in train_window(df_train, _mmodels, mt, mix=_mmix,
                                       wagi_klas=_mwagi).items():
                mix_bank[(_cfg, _m)] = _d
        log.info(f"    wytrenowano {len(bank)} czystych + {len(mix_bank)} mixowanych w {time.time()-t_tr:.0f}s")
        if not bank and not mix_bank:
            continue
        per_window_cutoffs.append(cutoff.strftime("%Y-%m-%d"))

        for ci, (cfg_name, (_models, _mx)) in enumerate(plan.items(), 1):
            # Czyste modele z banku; mixowane z mix_bank pod kluczem (config, model)
            avail = {m: bank[m] for m in _models if m in bank}
            if _mx:
                for _m in _models:
                    if _m in _mx and (cfg_name, _m) in mix_bank:
                        avail[_m] = mix_bank[(cfg_name, _m)]
            if not avail:
                continue
            inject(ensemble, avail)
            # Symulacja RÓWNOLEGLE po symbolach (2026-07-14). Wczesniej petla byla
            # sekwencyjna: 76 configow x 106 symboli x 12 okien, jeden symbol na raz
            # — na RunPodzie (256 rdzeni) dawalo to 96% BEZCZYNNEGO CPU i ~2h symulacji
            # na okno. Backtester i tak liczy kazdy symbol niezaleznie (wlasne swiece,
            # wlasne trade'y), wiec zrownoleglenie jest bezpieczne. Ensemble jest
            # WSPOLDZIELONY i tylko CZYTANY (predict) — inject() dzieje sie PRZED pula.
            trades = []
            _syms = backtester._list_symbols()

            def _sim(sym):
                c1h = backtester._load_window(sym, "1h", off_start, off_end)
                if not c1h:
                    return []
                c4h = backtester._load_window(sym, "4h", off_start, off_end)
                c1d = backtester._load_window(sym, "1d", off_start, off_end)
                try:
                    return backtester.run_simulation_ai(c1h, c4h, c1d, sym, args.mode)
                except Exception:
                    return []

            # SWEEP PROGOW: modele sa juz wytrenowane i wstrzykniete (inject wyzej),
            # a prog dziala dopiero w run_simulation_ai. Wiec dla kazdego progu
            # powtarzamy TYLKO symulacje — trening (najdrozszy krok) raz na okno.
            for _thr in (_SWEEP or [None]):
                if _thr is not None:
                    bt_mod._DECISION_THRESHOLD = _thr
                    _klucz = f"{cfg_name}@t{_thr:g}"
                else:
                    _klucz = cfg_name

                trades = []
                with ThreadPoolExecutor(max_workers=SIM_WORKERS) as ex:
                    for r in ex.map(_sim, _syms):
                        trades += r
                st = backtester._wfv_window_stats(trades)
                st["window"] = f"W{w+1}"
                per_window_results.setdefault(_klucz, []).append(st)
                save_window(run_id, _klucz, st, cutoff.strftime("%Y-%m-%d"))
                save_trades(run_id, _klucz, f"W{w+1}", trades)
                log.info(f"    [{ci}/{len(plan)}] {_klucz:28} trades={st.get('total_trades'):<5} "
                         f"pf={st.get('profit_factor')} sharpe={st.get('sharpe_ratio')}")

            # przywroc bazowy prog, zeby kolejny config nie odziedziczyl ostatniego ze sweepu
            if _SWEEP:
                bt_mod._DECISION_THRESHOLD = args.threshold

    # Shard liczy tylko czesc okien — zapisuje je jako czastkowe (window_days
    # ujemne = marker "czastkowy"), a werdykt liczy dopiero merge_shards().
    if shard_n > 1:
        part = ROOT / "wfv_shards" / f"{run_id}"
        part.mkdir(parents=True, exist_ok=True)
        (part / f"shard{shard_i}.json").write_text(json.dumps(
            {"run_id": run_id, "shard": shard_i, "cutoffs": per_window_cutoffs,
             "results": {k: v for k, v in per_window_results.items() if v},
             "args": {"threshold": args.threshold, "vote_gate": args.vote_gate,
                      "window_days": args.window_days, "mode": args.mode}},
            default=str))
        log.info(f"=== SHARD {shard_i}/{shard_n} GOTOWY | {part}/shard{shard_i}.json ===")
        return

    # werdykt per config z zebranych okien
    for cfg_name, windows in per_window_results.items():
        if not windows:
            log.warning(f"{cfg_name}: zero okien — brak wyniku")
            continue
        # Klucze sweepu maja postac "<config>@t<prog>" i NIE ma ich w `plan` —
        # modele bierzemy z configu bazowego (sweep nie zmienia modeli, tylko prog).
        _baza = cfg_name.split("@t")[0]
        models = plan[_baza][0]  # (models, mix) — wez czysta liste
        # Holdout firewall: split na dev (starsze okna) i holdout (N najnowszych).
        # W{n}: w=0=najstarsze okno -> holdout = najwyzsze numery okien.
        def _wnum(s):
            try: return int(str(s.get("window", "W0")).lstrip("W"))
            except Exception: return 0
        if args.holdout > 0:
            _cut = args.windows - args.holdout
            dev_windows = [w for w in windows if _wnum(w) <= _cut]
            hold_windows = [w for w in windows if _wnum(w) > _cut]
        else:
            dev_windows, hold_windows = windows, []
        # Decyzja GO/NO-GO liczona TYLKO z dev; holdout raportowany osobno jako firewall.
        v = backtester._wfv_verdict(dev_windows if dev_windows else windows)
        v_hold = backtester._wfv_verdict(hold_windows) if hold_windows else None
        rec = {
            "source_file": f"hai_wfv:{run_id}:{cfg_name}", "instance": "HAI_NL",
            "conf": getattr(args, "conf", None) or os.getenv("STRATEGY_MIN_CONFIDENCE", ""),
            "saved_at": datetime.now().isoformat(), "n_windows": len(dev_windows or windows),
            "window_days": args.window_days, "mode": args.mode,
            "voting_mode": "weighted", "decision_threshold": args.threshold,
            "vote_gate": args.vote_gate, "model_config": cfg_name,
            "models": ",".join(models),
            "avg_pf": v.get("avg_pf"), "min_pf": v.get("min_pf"),
            "min_pf_holdout": (v_hold.get("min_pf") if v_hold else None),
            "n_holdout": len(hold_windows),
            "max_dd": v.get("max_dd"), "avg_wr": v.get("avg_wr"),
            "avg_trades": v.get("avg_trades"), "weak_windows": v.get("weak_windows"),
            "decision": v.get("decision"), "harness": HARNESS, "lookahead_safe": 1,
            "sharpe": round(sum(w.get("sharpe_ratio") or 0 for w in windows) / len(windows), 3),
            "run_id": run_id, "train_cutoffs": ",".join(per_window_cutoffs),
            "ingested_at": datetime.now().isoformat(),
        }
        save_run(rec)
        _hold_str = ""
        if v_hold:
            _hm = v_hold.get("min_pf")
            _dm = v.get("min_pf")
            # firewall: holdout min_pf powinien trzymac poziom dev; duzy spadek = overfit lejka
            _flag = " ⚠️PEKA-NA-HOLDOUT" if (_hm is not None and _dm is not None and _hm < _dm * 0.7) else ""
            _hold_str = f" | HOLDOUT minpf={_hm} ({len(hold_windows)} okien){_flag}"
        log.info(f"{cfg_name:26} -> {v.get('decision'):8} pf={v.get('avg_pf')} "
                 f"minpf={v.get('min_pf')} dd={v.get('max_dd')} trades={v.get('avg_trades')}{_hold_str}")

    log.info(f"\n=== KONIEC | run {run_id} | baza: {DB} ===")
    return

    # (stara sciezka per-config — nieuzywana)
    for ci, (cfg_name, (_models, _mx)) in enumerate(plan.items(), 1):
        t_cfg = time.time()
        windows, cutoffs = [], []
        for w in range(args.windows):
            idx = args.windows - 1 - w
            off_end = idx * (args.window_days + args.embargo)
            off_start = off_end + args.window_days
            win_start = now - timedelta(days=off_start)
            cutoff = win_start - timedelta(days=args.embargo)

            df_train = df[df["timestamp"] < cutoff]
            if len(df_train) < 5000:
                log.warning(f"  W{w+1}: za mało danych treningowych ({len(df_train)}) — pomijam okno")
                continue

            log.info(f"  W{w+1}/{args.windows}: trening < {cutoff:%Y-%m-%d} "
                     f"({len(df_train):,} próbek) → test {win_start:%Y-%m-%d}..{(now-timedelta(days=off_end)):%Y-%m-%d}")
            trained = train_window(df_train, _models, mt, mix=_mx or None,
                                   wagi_klas=_wagi or None)
            if not trained:
                log.warning(f"  W{w+1}: zero modeli — okno pominięte")
                continue
            inject(ensemble, trained)
            cutoffs.append(cutoff.strftime("%Y-%m-%d"))

            trades = []
            for sym in backtester._list_symbols():
                c1h = backtester._load_window(sym, "1h", off_start, off_end)
                if not c1h:
                    continue
                c4h = backtester._load_window(sym, "4h", off_start, off_end)
                c1d = backtester._load_window(sym, "1d", off_start, off_end)
                try:
                    trades += backtester.run_simulation_ai(c1h, c4h, c1d, sym, args.mode)
                except Exception as e:
                    log.debug(f"    {sym}: {e}")
            st = backtester._wfv_window_stats(trades)
            st["window"] = f"W{w+1}"
            windows.append(st)
            log.info(f"  W{w+1}: trades={st.get('total_trades')} "
                     f"pf={st.get('profit_factor')} wr={st.get('win_rate')}")

        if not windows:
            log.warning(f"{cfg_name}: zero okien — brak wyniku")
            continue

        v = backtester._wfv_verdict(windows)
        rec = {
            "source_file": f"hai_wfv:{run_id}:{cfg_name}", "instance": "HAI_NL",
            "conf": getattr(args, "conf", None) or os.getenv("STRATEGY_MIN_CONFIDENCE", ""),
            "saved_at": datetime.now().isoformat(), "n_windows": len(windows),
            "window_days": args.window_days, "mode": args.mode,
            "voting_mode": "weighted", "decision_threshold": args.threshold,
            "vote_gate": args.vote_gate, "model_config": cfg_name,
            "models": ",".join(models),
            "avg_pf": v.get("avg_pf"), "min_pf": v.get("min_pf"),
            "max_dd": v.get("max_dd"), "avg_wr": v.get("avg_wr"),
            "avg_trades": v.get("avg_trades"), "weak_windows": v.get("weak_windows"),
            "decision": v.get("decision"), "harness": HARNESS, "lookahead_safe": 1,
            "run_id": run_id, "train_cutoffs": ",".join(cutoffs),
            "ingested_at": datetime.now().isoformat(),
        }
        save_run(rec)
        log.info(f"[{ci}/{len(plan)}] {cfg_name} -> {v.get('decision')} "
                 f"pf={v.get('avg_pf')} minpf={v.get('min_pf')} dd={v.get('max_dd')} "
                 f"trades={v.get('avg_trades')} | {time.time()-t_cfg:.0f}s → baza HAI-NL")

    log.info(f"\n=== KONIEC | run {run_id} | wyniki: sqlite3 {DB} "
             f'"select model_config,decision,avg_pf,min_pf from wfv_runs where run_id=\'{run_id}\'" ===')


if __name__ == "__main__":
    main()
