"""Cechy z wyzszych interwalow (4h, 1d) mapowane na swiece 1h — BEZ PRZECIEKU.

PRZECIEK, KTORY TO NAPRAWIA (wykryty 2026-09-13)
------------------------------------------------
Trening (ml_trainer) i walidator (backtester) mapowaly rsi_4h/trend_4h/rsi_1d/
trend_1d na godzine przez `searchsorted(czasy_otwarcia_tf, ts, side='right') - 1`.
To wybiera swiece 4h/1d, ktora sie ZACZELA <= ts, ale z wartoscia liczona na jej
KONCOWYM zamknieciu. Decyzja o 13:00 dostawala RSI ze swiecy 4h zamykajacej sie
o 16:00; dla 1d nawet 23 godziny przyszlosci.

Zmierzone na 8112 wejsciach walidatora (korelacja cechy z ruchem ceny):
    rsi_4h   przeszlosc +0.113   PRZYSZLOSC +0.693
    rsi_1d   przeszlosc +0.103   PRZYSZLOSC +0.474
    rsi 1h   przeszlosc +0.607   przyszlosc -0.155   <- tak wyglada uczciwa cecha
Kierunek ceny po 6h zgadzal sie w walidatorze w 92.1%, na zywo w 53.8%.

CO ROBI SILNIK NA ZYWO — I CO TU ODTWARZAMY
-------------------------------------------
engine pobiera z gieldy swiece 4h/1d (limit 300). Ostatnia jest W TOKU: jej
"close" to biezaca cena. features.build_features_live liczy na tym szeregu:
    rsi   = strategy.calculate_rsi(ceny, 14)  — SREDNIA PROSTA 14 ostatnich zmian
    trend = _trend_jak_trening(ceny)          — EMA(9) vs EMA(21), prog +-0.3%
Dla kazdej swiecy 1h budujemy wiec DOKLADNIE ten szereg: zamkniecia swiec tf,
ktore skonczyly sie przed biezacym przedzialem, plus biezace zamkniecie 1h jako
swieca w toku. Uwaga: trening i walidator liczyly RSI wygladzaniem Wildera, a
produkcja srednia prosta — dwa rozne wzory pod ta sama nazwa. Tu obowiazuje
wzor produkcji, bo to ona podejmuje decyzje.

Jeden plik, importowany przez OBA miejsca. Czterokrotnie w tej sesji poprawka
trafila tylko do jednej z dwoch rownoleglych implementacji.
"""
from __future__ import annotations

import numpy as np

# Progi minimalnej historii — te same co w features.py (MIN_PRICES_4H/1D).
MIN_4H = 20
MIN_1D = 15
RSI_OKRES = 14


def _ema_rek(x: np.ndarray, span: int) -> np.ndarray:
    """EMA jak pandas ewm(span, adjust=False): start od pierwszej wartosci."""
    k = 2.0 / (span + 1.0)
    out = np.empty(len(x), dtype=np.float64)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1.0 - k)
    return out


def rsi_sma(closes: np.ndarray, period: int = RSI_OKRES) -> np.ndarray:
    """RSI z 1h jak AIStrategy.calculate_rsi na zywo — dla KAZDEJ swiecy.

    Produkcja: srednia prosta z `period` ostatnich zmian (suma/period), 50 gdy
    za malo danych, 100 gdy w oknie nie ma ani jednej straty. Trening
    (ml_trainer.calc_rsi) i walidator (backtester._vec_rsi) liczyly to
    wygladzaniem Wildera — inna liczba pod ta sama nazwa `rsi` (2026-09-13).
    Zero strat sprawdzane licznikiem zmian ujemnych, nie roznica sum
    skumulowanych, bo ta daje 1e-17 zamiast zera i psuje przypadek 100.
    """
    c = np.asarray(closes, dtype=np.float64)
    n = len(c)
    out = np.full(n, 50.0)
    if n < period + 1:
        return out
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    cg = np.concatenate([[0.0], np.cumsum(g)])
    cl = np.concatenate([[0.0], np.cumsum(l)])
    cn = np.concatenate([[0], np.cumsum(d < 0)])
    i = np.arange(period, n)                 # swieca i: zmiany d[i-period .. i-1]
    sg = (cg[i] - cg[i - period]) / period
    sl = (cl[i] - cl[i - period]) / period
    ujemnych = cn[i] - cn[i - period]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ujemnych == 0, 100.0, 100.0 - 100.0 / (1.0 + sg / sl))
    out[period:] = r
    return out


def tf_w_toku(c1h: np.ndarray, t1h: np.ndarray,
              ctf: np.ndarray, ttf: np.ndarray, min_swiec: int):
    """rsi i trend interwalu tf dla kazdej swiecy 1h, liczone jak na zywo.

    c1h, t1h — zamkniecia i czasy OTWARCIA swiec 1h (ta sama jednostka co ttf)
    ctf, ttf — zamkniecia i czasy OTWARCIA swiec tf (4h albo 1d), rosnaco
    Zwraca (rsi, trend) o dlugosci len(c1h); brak historii -> 50.0 / 0.0.
    """
    n = len(c1h)
    rsi = np.full(n, 50.0)
    trend = np.zeros(n)
    if n == 0 or len(ctf) == 0:
        return rsi, trend
    c1h = np.asarray(c1h, dtype=np.float64)
    ctf = np.asarray(ctf, dtype=np.float64)

    # b = indeks swiecy tf, w ktorej lezy dana swieca 1h (otwarcie tf <= t1h).
    # Zamkniete przed nia sa ctf[0 .. b-1]; biezaca w toku ma close = c1h[i].
    b = np.searchsorted(np.asarray(ttf), np.asarray(t1h), side="right") - 1
    dl = b + 1                        # dlugosc szeregu, jaki widzi silnik

    # --- RSI: srednia prosta z 14 zmian = 13 zamknietych + 1 w toku ---
    d = np.diff(ctf)                  # d[j] = ctf[j+1] - ctf[j]
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    cg = np.concatenate([[0.0], np.cumsum(g)])
    cl = np.concatenate([[0.0], np.cumsum(l)])
    ok_rsi = (b >= RSI_OKRES) & (dl >= min_swiec)
    bi = np.where(ok_rsi, b, RSI_OKRES)
    # zmiany zamkniete: d[b-14 .. b-2] -> 13 sztuk
    sg = cg[bi - 1] - cg[bi - RSI_OKRES]
    sl = cl[bi - 1] - cl[bi - RSI_OKRES]
    ost = ctf[np.clip(bi - 1, 0, len(ctf) - 1)]
    pd_ = c1h - ost
    ag = (sg + np.maximum(pd_, 0.0)) / RSI_OKRES
    al = (sl + np.maximum(-pd_, 0.0)) / RSI_OKRES
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(al < 1e-10, 100.0, 100.0 - 100.0 / (1.0 + ag / al))
    rsi = np.where(ok_rsi, r, 50.0)

    # --- trend: EMA(9) vs EMA(21) na [zamkniete..., w toku] ---
    ef_full = _ema_rek(ctf, 9)
    es_full = _ema_rek(ctf, 21)
    ok_tr = (b >= 1) & (dl >= max(21, min_swiec))
    bj = np.where(ok_tr, b, 1)
    k9, k21 = 2.0 / 10.0, 2.0 / 22.0
    ef = c1h * k9 + ef_full[bj - 1] * (1.0 - k9)
    es = c1h * k21 + es_full[bj - 1] * (1.0 - k21)
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = np.where(es != 0, (ef - es) / es * 100.0, 0.0)
    tr = np.where(dp > 0.3, 1.0, np.where(dp < -0.3, -1.0, 0.0))
    trend = np.where(ok_tr, tr, 0.0)
    return rsi, trend
