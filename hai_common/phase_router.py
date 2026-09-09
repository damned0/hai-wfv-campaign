"""Router reżimowo-fazowy — jawna sztafeta modeli.

Na podstawie danych z NewHorizonts: system ma już dziś ukrytą rotację
(histgb_fast6h dominuje w korektach, histgb w hossie, cat_sharp6x w bessie).
Router formalizuje tę rotację: włącza/wyłącza modele w zależności od fazy rynkowej.

Fazy:
  BULL: trend wzrostowy — histgb+histgb_fast6h (dostawcy wolumenu hossy)
  BEAR: bessa — cat_sharp6x (+ short-specjaliści)
  CHOP: dryf/konsolidacja — redukcja ekspozycji (tylko najwyższy konsensus)
  MEAN_REV: korekta w trendzie — histgb_fast6h (sprinter korekt)
  HIGH_VOL: wysoka zmienność — wszystkie modele, ale podwyższony próg

Konfiguracja:
  PHASE_ROUTER=on|off (domyślnie off)
"""

import os
import logging
from typing import Dict, List, Optional

from .config import BASE_DIR
from .regime_detector import regime_detector

logger = logging.getLogger(__name__)


class PhaseRouter:
    ROLE_BULL = "histgb+histgb_fast6h"
    ROLE_BEAR = "cat_sharp6x+short_specialists"
    ROLE_CHOP = "reduced_exposure"
    ROLE_MEAN_REV = "histgb_fast6h_sprinter"
    ROLE_HIGH_VOL = "all_models_high_threshold"

    def __init__(self):
        self.enabled = os.getenv("PHASE_ROUTER", "off").lower() == "on"
        self.current_phase: str = "unknown"
        self.active_roles: List[str] = []
        self.threshold_multiplier: float = 1.0
        self._phases = {
            self.ROLE_BULL: [],
            self.ROLE_BEAR: [],
            self.ROLE_CHOP: [],
            self.ROLE_MEAN_REV: [],
            self.ROLE_HIGH_VOL: [],
        }

    def detect_phase(self, btc_trend: str = None, regime: int = None,
                     vol_regime: str = None) -> str:
        if not self.enabled:
            return "passive"

        # Określ fazę na podstawie reżimu i trendu BTC
        if regime is None:
            try:
                regime = regime_detector.current_regime
            except Exception:
                regime = -1

        # Regime: -1=unknown, 0=trend, 1=mean_rev, 2=high_vol
        if regime == 1:  # mean_rev
            phase = self.ROLE_MEAN_REV
        elif regime == 2:  # high_vol
            phase = self.ROLE_HIGH_VOL
        elif btc_trend == "up":
            phase = self.ROLE_BULL
        elif btc_trend == "down":
            phase = self.ROLE_BEAR
        else:
            phase = self.ROLE_CHOP

        self.current_phase = phase
        self._update_roles(phase)
        return phase

    def _update_roles(self, phase: str):
        phase_roles = {
            self.ROLE_BULL: [self.ROLE_BULL],
            self.ROLE_BEAR: [self.ROLE_BEAR],
            self.ROLE_CHOP: [self.ROLE_CHOP],
            self.ROLE_MEAN_REV: [self.ROLE_MEAN_REV, self.ROLE_BULL],
            self.ROLE_HIGH_VOL: [self.ROLE_HIGH_VOL],
        }
        self.active_roles = phase_roles.get(phase, [self.ROLE_CHOP])
        self.threshold_multiplier = {
            self.ROLE_BULL: 1.0,
            self.ROLE_BEAR: 1.0,
            self.ROLE_CHOP: 1.3,
            self.ROLE_MEAN_REV: 0.9,
            self.ROLE_HIGH_VOL: 1.2,
        }.get(phase, 1.0)

    def should_trade(self, consensus_depth: int = 3) -> bool:
        if not self.enabled:
            return True
        if self.current_phase == self.ROLE_CHOP:
            return consensus_depth >= 4
        return consensus_depth >= 3

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "current_phase": self.current_phase,
            "active_roles": self.active_roles,
            "threshold_multiplier": self.threshold_multiplier,
        }


phase_router = PhaseRouter()
