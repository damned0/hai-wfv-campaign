# ===========================================
# HAI_EPV Engine ver.10 Final — core/ensemble.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje:
# - NeuralTraderEnsemble - ensemble drzew (LGB/RF/XGB/CAT/HistGB + specjalisci),
#   do 10 slotow (MAX_ENSEMBLE_MODELS), kazdy model wlasny zestaw cech
# - load_models()/_load_one() - dynamiczne skanowanie data/models/*.pkl
# - _recalc_weights() - wazenie glosow precision-basis
# - predict() - 3-class output NEUTRAL=0/LONG=1/SHORT=2, wazona suma prawdopodobienstw
# - status() - stan ensemble (modele/wagi/accuracy/cache) do dashboardu
# ===========================================
import hashlib
import json
import logging
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="lightgbm")
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import sys
import joblib
import numpy as np

# Propaguje do joblib/multiprocessing workers (RF n_jobs)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

# Compatibility shim: pkl files trained before core/torch_wrapper.py existed
# were serialized as __main__.TorchWrapper / __main__.MLP etc.
# Injecting the classes into __main__ lets joblib find them without retraining.
try:
    from . import torch_wrapper as _tw
    _main = sys.modules.get("__main__")
    if _main is not None:
        for _cls_name in ("TorchWrapper", "MLP", "LSTM", "TCNBlock", "TCN", "TransformerClassifier"):
            if not hasattr(_main, _cls_name):
                setattr(_main, _cls_name, getattr(_tw, _cls_name))
    del _tw, _main, _cls_name
except Exception:
    pass

logger = logging.getLogger(__name__)

try:
    from .config import BASE_DIR
    MODELS_DIR = BASE_DIR / "data" / "models"
except Exception:
    MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
# Limit czlonkow ensemble (audyt 2026-07-04) - budzet VPS: 12GB RAM/6 rdzeni/
# 50GB swap SSD, dzielony miedzy 5 instancji (EPV/DEV/LAB/LIV/TST). Dla
# samej INFERENCJI (predict_proba na juz zaladowanych modelach drzew) koszt
# pamieciowy per model to ~10-150MB (RF najwiekszy) - 10 modeli to <1.5GB,
# bezpieczne nawet gdy wszystkie 5 instancji dziala rownolegle. Prawdziwym
# ograniczeniem jest TRENING (RF ~5-7GB przejsciowo), nie liczba zaladowanych
# modeli do glosowania - stad limit tu jest hojny (10), a ostroznosc przy
# treningu (nie odpalac kilku ciezkich treningow rownolegle).
MAX_ENSEMBLE_MODELS = 10

# ── PARYTET Z WALIDATOREM (2026-08-13) ───────────────────────────────────────
# Te dwa progi byly ZASZYTE w predict() (0.33 i 0.30), podczas gdy walidator
# (backtester._VOTE_GATE / _DECISION_THRESHOLD) ma je parametryzowalne i
# hai_wfv.py ustawia je z CLI (--vote-gate / --threshold). Skutek: kampania
# WFV liczyla ENS-3x-diff przy vote_gate=0.40 i threshold=0.50, a produkcja
# jechala na 0.33/0.30 — czyli werdykt GO opisywal INNY uklad niz uruchomiony.
#
# Domyslne wartosci = DOTYCHCZASOWE ZACHOWANIE LIVE, zeby sama ta zmiana
# niczego nie przestawila. Zmiana nastepuje dopiero po jawnym ustawieniu env.
#   HAI_VOTE_GATE        <-> backtester._VOTE_GATE        (glos pojedynczego modelu)
#   HAI_DECISION_THRESHOLD <-> backtester._DECISION_THRESHOLD (zagregowany wynik)
def _prog(nazwa: str, domyslny: float) -> float:
    try:
        return float(os.environ.get(nazwa, domyslny))
    except (TypeError, ValueError):
        logger.warning("%s ma niepoprawna wartosc — uzywam %.2f", nazwa, domyslny)
        return domyslny


VOTE_GATE = _prog("HAI_VOTE_GATE", 0.33)              # modele 3-klasowe
VOTE_GATE_BIN_HI = _prog("HAI_VOTE_GATE_BIN_HI", 0.52)  # modele binarne: LONG
VOTE_GATE_BIN_LO = _prog("HAI_VOTE_GATE_BIN_LO", 0.48)  # modele binarne: SHORT
DECISION_THRESHOLD = _prog("HAI_DECISION_THRESHOLD", 0.30)

# audyt 2026-07-05 (v3, rozszerzona proba): mnozniki policzone z per-model
# per-regime PF w POLACZONYM trade_log WSZYSTKICH 7 dostepnych backtestow
# (14010 transakcji z przypisanym regime+dominant_model - znaczaco wieksza
# proba niz v2 na 2 plikach/3830tr). mult = clip(PF_modelu_w_reżimie /
# PF_sredni_w_reżimie, 0.7, 1.4), n<15 -> 1.0 (neutralnie z braku danych).
# ZASTRZEZENIE METODOLOGICZNE: te 7 backtestow to w wiekszosci ten sam
# 365-dniowy okres historyczny (rozne konfiguracje modeli patrzace na TEN
# SAM rynek), nie 7 niezaleznych prob - wieksza liczba wierszy nie oznacza
# proporcjonalnie wiecej niezaleznej informacji, tylko wiecej obserwacji
# tego samego okresu z roznych katow. Nadal lepsze niz 1 plik, ale nie
# traktowac jak WFV (ktore faktycznie testuje rozne okna czasowe).
#
# WAZNE: regime=0 (trend_following) ma ZERO transakcji we WSZYSTKICH 7
# backtestach - to bezposrednia konsekwencja INNEJ znalezionej dzisiaj luki:
# core/backtester.py CALKOWICIE BLOKUJE regime=0 (_regime_ok = regime_arr!=0),
# podczas gdy ai_strategy.py (live) tylko podnosi prog (+2%), nie blokuje.
# Brak danych = mnozniki dla regime=0 zostaja NEUTRALNE (1.0) dla wszystkich -
# nie "sprawdzone i neutralne", tylko niemozliwe do skalibrowania dopoki ta
# rozbieznosc backtester/live nie zostanie rozwiazana.
# RF ma tez za malo prob w kazdym reżimie (0-4 transakcji jako dominant_model
# we wszystkich 3, nawet na 14010 lacznie) - strukturalnie rzadko wygrywa
# argmax mimo niskiej korelacji glosu - zostawiony neutralny (1.0) wszedzie.
#
# UWAGA DLA UZYTKOWNIKA: ta zmiana wplywa TYLKO na live/paper trading.
# core/backtester.py nie wywoluje _regime_blended_weights() w ogole - kolejny
# odpalony stad backtest da IDENTYCZNY wynik jak przed ta zmiana, dopoki
# regime-blending nie zostanie wpiety do backtestera osobno.
REGIME_WEIGHTS = {
    0: {"lgb": 1.0, "rf": 1.0, "xgb": 1.0, "cat": 1.0, "histgb": 1.0,
        "lgb_fast24h": 1.0, "cat_sharp6x": 1.0},  # trend_following  → BRAK DANYCH (bt blokuje regime=0)
    1: {"lgb": 1.30, "rf": 1.0, "xgb": 0.70, "cat": 0.70, "histgb": 0.76,
        "lgb_fast24h": 0.85, "cat_sharp6x": 1.40},  # mean_reversion  → cat_sharp6x/lgb najlepsze (n=213-3489)
    2: {"lgb": 1.14, "rf": 1.0, "xgb": 0.99, "cat": 1.08, "histgb": 1.06,
        "lgb_fast24h": 1.04, "cat_sharp6x": 0.70},  # high_volatility → lgb najlepszy, cat_sharp6x najgorszy (n=118-2828)
}

# core_v3_profile "low" (audyt 2026-07-11, WFV na AD1-3): jedyny profil z GO
# na 6x90 (avg_pf=2.17 min_pf=1.43 dd=5.4% na CORE-v2-rf24h) i WARNING blisko
# GO na 12x45 (avg_pf=5.22 min_pf=1.00 dd=4.9%). Testowany dotad TYLKO w WFV
# (routes/trading.py na podach, core_v3_profile query param, per-request
# override) - tu wpiety NA STALE do bazowego REGIME_WEIGHTS zeby dzialal
# rowniez w live/paper (ai_strategy.py wywoluje ensemble.predict(regime=...)
# bezwarunkowo, _regime_blended_weights juz czyta ten slownik). Dotyczy
# WYLACZNIE modeli z kluczem tutaj (rf_fast24h zostaje neutralne 1.0,
# tak jak w oryginalnym tescie WFV - profil "low" mnozyl tylko ten jeden
# model). Zero wplywu na instancje ktore nie ladujaz et_h72_COREv2_iter2
# (np. EPV na dzien pisania tego kodu, config GP-div-h48).
REGIME_WEIGHTS[0]["et_h72_COREv2_iter2"] = 1.0
REGIME_WEIGHTS[1]["et_h72_COREv2_iter2"] = 0.3
REGIME_WEIGHTS[2]["et_h72_COREv2_iter2"] = 0.5

REGIME_NAMES = {0: "trend_following", 1: "mean_reversion", 2: "high_volatility"}

# Per-side (LONG/SHORT) model weight adjustment (audyt 2026-07-06, "Opcja C"
# z nocnej analizy LONG vs SHORT). histgb/histgb_fast6h maja swietny SHORT
# (PF do 13.84 w WFV tej nocy) ale wyraznie slabszy LONG (WR~41%, ważony
# po wszystkich 6 oknach WFV, najgorszy z przetestowanych rodzin modeli -
# kocur-family mial ~52.5%). Jeden wspolny mnoznik wagi dla obu kierunkow
# to architektoniczna niedorobka - ten dict pozwala przyciac wplyw modelu
# W JEDNYM kierunku bez dotykania drugiego (ktory dziala dobrze). Brak
# wpisu dla modelu/strony = mnoznik 1.0 (bez zmiany zachowania).
# Pierwsza, konserwatywna kalibracja - do zweryfikowania/dostrojenia po
# WFV. Tylko REDUKUJEMY zidentyfikowany słaby punkt, nie podbijamy SHORT
# ponad 1.0 bez dodatkowej kalibracji na realnym trade_logu.
# v2 (audyt 2026-07-06, po Test 15): long=0.5 na 6-modelowym kocur+ET dal
# ZERO mierzalnego efektu (rf/rf_h72/et i tak prawie nie glosuja, histgb
# waga juz rozcienczona miedzy 6 modeli). Zaostrzone do 0.1 i przeniesione
# do testu na 5-modelowym zestawie (Test 6/12/13) gdzie histgb ma wieksza
# wzgledna wage (~42% lacznie z histgb_fast6h vs ~32% w kocur+ET).
SIDE_WEIGHTS = {
    "histgb":        {"long": 0.1},
    "histgb_fast6h": {"long": 0.1},
}

FEATURE_NAMES = [
    # Przyciete do 19 cech (audyt 2026-07-03) — patrz ml_trainer.py po uzasadnienie
    "rsi", "rsi_4h", "rsi_1d",
    "ema_slow_r", "ema_mid_r",
    "atr_pct",
    "trend_4h", "trend_1d",
    "funding_rate",
    "oi_change_24h", "oi_zscore_30d",
    "price_position_bb", "bb_bandwidth_pct",
    "hour_sin", "hour_cos", "day_of_week",
    "adx_14",
    "macd_hist", "sr_node_strength",
]

@dataclass
class ModelThought:
    model_name: str
    prediction: str
    probability: float
    confidence: float
    weight: float
    reason: str

class NeuralTraderEnsemble:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.accuracies = {}
        self.f1_scores = {}
        self.precision_scores = {}
        self.weights = {}
        self.feature_names: Dict[str, list] = {}
        self.active = False
        self._cache = OrderedDict()
        self._cache_maxsize = 8192
        self._cache_hits = 0
        self._cache_misses = 0
        # Vote-labeling post-hoc (audyt 2026-07-05) - lazy-load, False = "juz
        # probowane wczytac, brak pliku" (odroznia od None = "jeszcze nie probowane")
        self._meta_label = None
        self._confidence_calib = None

    def _get_meta_label(self):
        if self._meta_label is None:
            from .meta_label import load_meta_label
            self._meta_label = load_meta_label() or False
        return self._meta_label or None

    def _get_confidence_calib(self):
        if self._confidence_calib is None:
            from .confidence_calib import load_calibration
            self._confidence_calib = load_calibration() or False
        return self._confidence_calib or None

    # Solo-backtest PF (audyt 2026-07-04) - PRZETESTOWANE jako baza wagi
    # ensemble, WYNIK GORSZY niz precision-basis (PF 1.36 vs 1.39 w pelnym
    # backtescie) mimo ze CAT mial najgorszy solo-PF (1.00, strata) a
    # najlepsza precyzje walidacyjna (0.55). Wniosek: CAT mimo slabego SOLO
    # wynosi cos wartosciowego DO BLENDU (koryguje bledy pozostalych 4 w
    # niektorych przypadkach) - wyzerowanie go zaszkodzilo. Zostawione w
    # kodzie jako udokumentowany, odrzucony eksperyment (nie usuwamy -
    # przyda sie gdy ktos znow zapyta "a moze wazyc po realnym PF?").
    _SOLO_BACKTEST_PF = {
        'lgb': 1.15, 'rf': 1.15, 'xgb': 1.14, 'histgb': 1.14, 'cat': 1.00,
    }
    _USE_SOLO_PF_WEIGHTING = False  # przetestowane 2026-07-04, gorszy wynik - zostaje wylaczone

    def _recalc_weights(self):
        if not self.models:
            self.weights = {}
            return
        if self._USE_SOLO_PF_WEIGHTING:
            edges = {m: max(self._SOLO_BACKTEST_PF.get(m, 1.0) - 1.0, 0.0) for m in self.models}
            total_edge = sum(edges.values())
            if total_edge > 0:
                self.weights = {m: round(e / total_edge, 4) for m, e in edges.items()}
                logger.info(f"Ensemble wagi (solo-backtest PF basis): {self.weights}")
                return
        # Precision-basis - sprawdzony, lepszy wynik (PF 1.39 vs 1.36 solo-PF).
        scores = {
            m: self.precision_scores.get(m) or self.f1_scores.get(m) or self.accuracies.get(m, 0.10)
            for m in self.models
        }
        total = sum(scores.values())
        if total <= 0:
            n = max(len(self.models), 1)
            self.weights = {m: round(1 / n, 4) for m in self.models}
        else:
            self.weights = {m: round(s / total, 4) for m, s in scores.items()}
        logger.info(f"Ensemble wagi (precision basis, fallback f1->accuracy): {self.weights}")

    def _load_one(self, name: str) -> bool:
        """Laduje POJEDYNCZY model z dysku (audyt 2026-07-04 - wydzielone z
        load_models() zeby /ctrl/model-toggle mogl przywrocic jeden model bez
        przeladowywania calej piatki). Zwraca True jesli sie udalo."""
        try:
            path = MODELS_DIR / f"{name}.pkl"
            if not path.exists():
                logger.warning(f"Brak modelu: {path}")
                return False
            data = joblib.load(path)
            self.models[name] = data["model"]
            self.scalers[name] = data.get("scaler")
            self.accuracies[name] = data.get("accuracy", 0.0)
            self.f1_scores[name] = data.get("f1") or data.get("f1_long", 0.0)
            self.precision_scores[name] = data.get("precision") or data.get("wf_precision") or 0.0
            self.feature_names[name] = data.get("features") or data.get("feature_names", FEATURE_NAMES)
            logger.info(f"✓ {name.upper()} loaded | acc={data.get('accuracy', 0):.2%} "
                        f"f1={self.f1_scores[name]:.3f} | features={len(self.feature_names[name])}")
            return True
        except Exception as e:
            logger.warning(f"Nie załadowano {name}: {e}")
            return False

    def _apply_model_config(self):
        """Wspolna baza modeli (audyt 2026-07-07, na prosbe usera - "zestawy w
        folderach konfiguracji... wstawiaj tylko linki... wspolna baza... i
        HAIresearch"). Zamiast fizycznego kopiowania .pkl miedzy instancjami
        (dotychczasowa metoda - podatna na rozjazdy, kazda instancja ma wlasna
        kopie), nazwany config w model_configs/{NAME}.json wskazuje liste
        modeli, a ich JEDYNA kopia zyje w model_registry/ (wspolna, poza
        instancjami) - tu tworzymy tylko SYMLINKI. Aktywacja: zmienna
        HAI_MODEL_CONFIG (np. "CATalpha") w .env instancji. Brak zmiennej =
        stare zachowanie (skan katalogu), pelna wsteczna kompatybilnosc.
        Ten sam model_configs/model_registry beda zrodlem dla HAIresearch."""
        cfg_name = os.getenv("HAI_MODEL_CONFIG")
        if not cfg_name:
            return
        _root = Path(os.environ.get("HAI_ROOT", "/root/ProjektHAI"))
        registry = _root / "model_registry"
        config_path = _root / "model_configs" / f"{cfg_name}.json"
        if not config_path.exists():
            logger.warning(f"HAI_MODEL_CONFIG={cfg_name} ale brak {config_path}")
            return
        try:
            cfg = json.loads(config_path.read_text())
            wanted = set(cfg.get("models", []))

            # Sprzatanie symlinkow z POPRZEDNIEGO configu (bug znaleziony
            # 2026-07-07: przelaczenie K02->K03 zostawilo cat_fast6h/
            # histgb_fast6h/rf_h72 jako symlinki z K02, ktore skan katalogu
            # w load_models() dalej podnosil jako aktywne, mimo ze nie byly
            # w nowym configu). Usuwamy TYLKO symlinki wskazujace do
            # model_registry/ i NIE bedace czescia biezacego configu -
            # prawdziwe .pkl (nie-symlinki) nigdy nie ruszane.
            for existing in MODELS_DIR.glob("*.pkl"):
                if existing.stem in wanted:
                    continue
                if existing.is_symlink():
                    try:
                        target = existing.resolve()
                        reg = registry.resolve()
                        # 2026-08-03: byla rownosc target.parent==reg, zakladajaca
                        # plaski model_registry/xxx.pkl. Rejestr jest teraz
                        # zagniezdzony (model_registry/gen.SNPR/6h/xxx_6h.pkl),
                        # wiec target.parent to .../gen.SNPR/6h - nigdy rowne reg,
                        # warunek nigdy nie byl prawdziwy i osierocone symlinki z
                        # poprzednich configow nigdy nie byly usuwane (potwierdzone:
                        # EPV mial xgb_rev.pkl, LAB mial cat_sniper.pkl jako smieci
                        # sprzed dnia+, oba instancje glosowaly blendem zamiast
                        # jednym modelem z configu). Sprawdzamy przodka, nie
                        # bezposredniego rodzica.
                        if reg == target or reg in target.parents:
                            existing.unlink()
                    except OSError:
                        pass

            for model_name in cfg.get("models", []):
                dst = MODELS_DIR / f"{model_name}.pkl"
                # Kanoniczna nazwa w configu: <algo>_<profil>_<H>[.<variant>] (np. cat_sniper_6h, et_sniper_6h.v1).
                # Zrodlo: plaski model_registry/<name>.pkl albo zagniezdzony model_registry/gen.SNPR/<H>/...
                _variant = None
                if "." in model_name:
                    model_name, _variant = model_name.split(".", 1)
                src = registry / f"{model_name}.pkl"
                if not src.exists():
                    _h = None
                    for _part in reversed(model_name.split("_")):
                        _digits = "".join(ch for ch in _part if ch.isdigit())
                        if _digits:
                            _h = int(_digits)
                            break
                    _base = model_name
                    if _h is not None:
                        _tokens = model_name.split("_")
                        while _tokens and any(ch.isdigit() for ch in _tokens[-1]):
                            _tokens.pop()
                        _base = "_".join(_tokens)
                    _base = _base.lower()
                    _hor = _h or 6
                    # 1) gen.SNPR/<H>/<base>_<H>.<variant>.pkl  (warianty, np. et_sniper_6h.v1)
                    if _variant:
                        _cand = registry / "gen.SNPR" / f"{_hor}h" / f"{_base}_{_hor}h.{_variant}.pkl"
                        if _cand.exists():
                            src = _cand
                    # 2) gen.SNPR/<H>/<base>_<H>.pkl
                    if not src.exists():
                        _cand = registry / "gen.SNPR" / f"{_hor}h" / f"{_base}_{_hor}h.pkl"
                        if _cand.exists():
                            src = _cand
                    # 3) flat registry/<base>.<variant>.pkl (stare warianty, np. xgb_sniper_deploy.pkl)
                    if not src.exists() and _variant:
                        _cand = registry / f"{_base}_{_variant}.pkl"
                        if _cand.exists():
                            src = _cand
                if not src.exists():
                    logger.warning(f"model_registry brak {model_name}.pkl dla configu {cfg_name}")
                    continue
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src)
            logger.info(f"Model config '{cfg_name}' zastosowany: {cfg.get('models')}")
        except Exception as e:
            logger.error(f"Blad stosowania model config '{cfg_name}': {e}")

    def load_models(self):
        self._apply_model_config()
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.models = {}
        self.scalers = {}
        self.accuracies = {}
        self.f1_scores = {}
        self.feature_names = {}
        # audyt 2026-07-04: dynamiczne skanowanie MODELS_DIR zamiast sztywnej
        # listy 5 nazw - kazdy .pkl w folderze (poza _NEW/_backup, ktore sa
        # miejscem stagingu/kopii zapasowych, nie aktywnymi czlonkami
        # ensemble) staje sie automatycznie czlonkiem glosujacym, do limitu
        # MAX_ENSEMBLE_MODELS. Pozwala dokladac warianty (np. rf_4h.pkl)
        # samym skopiowaniem pliku, bez zmiany kodu.
        # regime_hmm to HMM detektor rezimu, meta_label/confidence_calib to
        # post-hoc filtry vote-labelingu (audyt 2026-07-05) - ZADEN z nich
        # nie jest klasyfikatorem 3-klasowym, maja inna strukture pliku
        # (dict {model,scaler,feature_cols,...} zamiast bezposredniego
        # klasyfikatora) - BUG znaleziony po restarcie: bez tego wpisu
        # meta_label.pkl wskakiwal jako "model glosujacy" do ensemble.
        _NOT_ENSEMBLE_MEMBERS = {"regime_hmm", "meta_label", "confidence_calib"}
        pkl_files = sorted(MODELS_DIR.glob("*.pkl"))
        loaded = 0
        for path in pkl_files:
            name = path.stem
            if name in _NOT_ENSEMBLE_MEMBERS:
                continue
            if name.endswith("_NEW") or name.endswith("_backup") or name.startswith("backup_"):
                continue
            if loaded >= MAX_ENSEMBLE_MODELS:
                logger.warning(f"Limit {MAX_ENSEMBLE_MODELS} modeli osiagniety, pomijam: {name}")
                break
            try:
                data = joblib.load(path)
                self.models[name] = data["model"]
                self.scalers[name] = data.get("scaler")
                self.accuracies[name] = data.get("accuracy", 0.0)
                self.f1_scores[name] = data.get("f1") or data.get("f1_long", 0.0)
                self.precision_scores[name] = data.get("precision") or data.get("wf_precision") or 0.0
                # RF zapisuje listę cech pod kluczem "features", LGB/XGB (nowszy
                # skrypt treningowy) pod "feature_names" — bez tego LGB/XGB dostawały
                # domyślny FEATURE_NAMES (złej długości), scaler rzucał ValueError,
                # łapane cicho przez except poniżej — oba modele zawsze martwe.
                self.feature_names[name] = data.get("features") or data.get("feature_names", FEATURE_NAMES)
                logger.info(f"✓ {name.upper()} loaded | acc={data.get('accuracy', 0):.2%} "
                            f"f1={self.f1_scores[name]:.3f} | features={len(self.feature_names[name])}")
                loaded += 1
            except Exception as e:
                logger.warning(f"Nie załadowano {name}: {e}")
        # MLP/LSTM/TCN usuniete z EPV (audyt 2026-07-03): ablacja pokazala ze
        # ich glosy obnizaly PF calego ensemble (PF 1.15->1.19 po usunieciu w
        # zwyklym backtescie, WFV avg_pf 1.145->1.172, weak_windows 3->2).
        # Kod architektur zostaje wylacznie w TST (wylaczony z aktywnej listy).
        self.active = len(self.models) >= 1  # 2026-08-01: solo modele tez aktywne
        self._recalc_weights()
        logger.info(f"Ensemble ver.10 active: {self.active} | models: {list(self.models.keys())} | wagi: {self.weights}")
        try:
            from .state import state
            state.add_log("ai", "INFO", event="MODEL_LOAD", model_name="ensemble",
                          message=f"Zaladowano {len(self.models)} modeli: {list(self.models.keys())} | wagi: {self.weights}")
        except Exception:
            pass

    def _regime_blended_weights(self, regime: int) -> Dict[str, float]:
        """Wagi accuracy przemnożone przez mnożniki reżimu, renormalizowane."""
        mults = REGIME_WEIGHTS.get(regime, {})
        if not mults:
            return self.weights
        blended = {k: self.weights.get(k, 0.0) * mults.get(k, 1.0) for k in self.weights}
        total = sum(blended.values()) or 1.0
        return {k: round(v / total, 4) for k, v in blended.items()}

    def predict(self, features: Dict, regime: int = None, seq: "np.ndarray | None" = None) -> Dict:
        """Predict action from feature dict.

        seq: optional (seq_len, n_feat) float32 array for LSTM/TCN models.
             Built by build_feature_sequence_live() in features.py.
             If None, LSTM/TCN models are skipped.
        """
        if not self.active:
            return {"action": "NEUTRAL", "confidence": 0.0, "reason": "Ensemble not ready"}

        if regime is None:
            try:
                from .regime_detector import regime_detector as _rd
                if _rd.is_trained:
                    regime = _rd.current_regime
            except Exception:
                pass

        fhash = hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()
        seq_hash = hashlib.md5(seq.tobytes()).hexdigest() if seq is not None else "noseq"
        cache_key = f"{fhash}:{regime}:{seq_hash}"
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1

        try:
            X_raw = np.array([[float(features.get(f, 0.0)) for f in FEATURE_NAMES]], dtype=np.float64)
        except (TypeError, ValueError) as e:
            logger.warning(f"Bad features for ensemble: {e}")
            return {"action": "NEUTRAL", "confidence": 0.0, "reason": "Bad features"}

        effective_weights = self._regime_blended_weights(regime) if regime is not None else self.weights
        regime_name = REGIME_NAMES.get(regime, "unknown") if regime is not None else "unknown"

        from .config import config as _cfg
        _calib = self._get_confidence_calib() if _cfg.CONFIDENCE_CALIB_ENABLED else None

        long_score = short_score = 0.0
        thoughts = []
        raw_votes: Dict[str, Dict[str, float]] = {}
        for name, model in self.models.items():
            try:
                model_features = self.feature_names.get(name, FEATURE_NAMES)
                # Neural sequence models need a (1, seq_len, n_feat) input
                is_seq_model = hasattr(model, "model_type") and model.model_type in ("lstm", "tcn", "transformer_cls")
                if is_seq_model:
                    if seq is None:
                        continue  # can't run without sequence
                    X_input = seq[np.newaxis].astype(np.float32)  # (1, seq_len, n_feat)
                else:
                    if model_features != FEATURE_NAMES:
                        X_input = np.array([[float(features.get(f, 0.0)) for f in model_features]], dtype=np.float64)
                    else:
                        X_input = X_raw
                    sc = self.scalers.get(name)
                    X_input = sc.transform(X_input) if sc is not None else X_input

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    proba = model.predict_proba(X_input)[0]

                w = effective_weights.get(name, 1.0 / max(len(self.models), 1))

                if len(proba) == 3:
                    # 3-class neural: [P_neutral, P_long, P_short]
                    lp = float(proba[1])
                    sp = float(proba[2])
                    if _calib is not None:
                        from .confidence_calib import calibrate
                        lp = calibrate(_calib, name, lp)
                        sp = calibrate(_calib, name, sp)
                    raw_votes[name] = {"long": lp, "short": sp}
                    if lp > sp and lp > VOTE_GATE:
                        pred = "LONG"
                        long_score  += lp * w * SIDE_WEIGHTS.get(name, {}).get("long", 1.0)
                    elif sp > lp and sp > VOTE_GATE:
                        pred = "SHORT"
                        short_score += sp * w * SIDE_WEIGHTS.get(name, {}).get("short", 1.0)
                    else:
                        pred = "NEUTRAL"
                    thoughts.append(ModelThought(
                        model_name=name, prediction=pred,
                        probability=round(max(lp, sp), 4),
                        confidence=round(max(lp, sp), 4), weight=round(w, 4),
                        reason=f"{pred} | L={lp:.1%} S={sp:.1%} | w={w:.3f} | acc={self.accuracies.get(name, 0):.1%}",
                    ))
                else:
                    # Binary tree model: [P_notlong, P_long]
                    lp = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    if _calib is not None:
                        from .confidence_calib import calibrate
                        lp = calibrate(_calib, name, lp)
                    raw_votes[name] = {"long": lp, "short": 1.0 - lp}
                    if lp > VOTE_GATE_BIN_HI:
                        pred = "LONG"
                        long_score += lp * w * SIDE_WEIGHTS.get(name, {}).get("long", 1.0)
                    elif lp < VOTE_GATE_BIN_LO:
                        pred = "SHORT"
                        short_score += (1 - lp) * w * SIDE_WEIGHTS.get(name, {}).get("short", 1.0)
                    else:
                        pred = "NEUTRAL"
                    thoughts.append(ModelThought(
                        model_name=name, prediction=pred,
                        probability=round(lp, 4),
                        confidence=round(max(lp, 1 - lp), 4), weight=round(w, 4),
                        reason=f"{pred} | prob={lp:.1%} | w={w:.3f} | acc={self.accuracies.get(name, 0):.1%} | regime={regime_name}",
                    ))
            except Exception as e:
                logger.debug(f"Model {name} error: {e}")

        if long_score > DECISION_THRESHOLD and long_score > short_score:
            action = "LONG"
        elif short_score > DECISION_THRESHOLD and short_score > long_score:
            action = "SHORT"
        else:
            action = "NEUTRAL"

        meta_score = None
        if action != "NEUTRAL" and _cfg.META_LABEL_ENABLED:
            meta = self._get_meta_label()
            if meta is not None:
                try:
                    from .meta_label import score_trade
                    meta_score = score_trade(
                        meta, side=action, confidence=max(long_score, short_score),
                        model_votes=raw_votes, bb_pos=features.get("price_position_bb", 0.5),
                        regime=regime, feature_snapshot=features,
                    )
                    if meta_score < _cfg.META_LABEL_THRESHOLD:
                        action = "NEUTRAL"
                except Exception as e:
                    logger.debug(f"meta_label score error: {e}")

        result = {
            "action": action,
            "confidence": round(max(long_score, short_score), 4),
            "long_score": round(long_score, 4),
            "short_score": round(short_score, 4),
            "thoughts": [t.__dict__ for t in thoughts],
            "meta_score": round(meta_score, 4) if meta_score is not None else None,
            "regime": regime,
            "regime_name": regime_name,
            "reason": f"Ensemble {action} | L={long_score:.3f} S={short_score:.3f} | regime={regime_name}",
        }
        if len(self._cache) >= self._cache_maxsize:
            self._cache.popitem(last=False)
        self._cache[cache_key] = result
        return result

    def status(self) -> Dict:
        cache_total = self._cache_hits + self._cache_misses
        cache_rate = (self._cache_hits / cache_total * 100) if cache_total > 0 else 0
        return {
            "active": self.active,
            "models": list(self.models.keys()),
            "model_count": f"{len(self.models)}/{MAX_ENSEMBLE_MODELS}",
            "f1_scores": {k: round(v, 4) for k, v in self.f1_scores.items()},
            "accuracies": {k: round(v, 4) for k, v in self.accuracies.items()},
            "feature_counts": {k: len(v) for k, v in self.feature_names.items()},
            "weights": self.weights,
            "cache": {"size": len(self._cache), "maxsize": self._cache_maxsize,
                      "hits": self._cache_hits, "misses": self._cache_misses,
                      "hit_rate": f"{cache_rate:.1f}%"},
        }

    def get_thoughts(self, features: Dict):
        if not self.active:
            self.load_models()
        return self.predict(features).get("thoughts", [])

ensemble = NeuralTraderEnsemble()
