"""
Bounded Adaptive Ventilation Controller.
Implements clinical safety constraints (ARDSNet / Lung-Protective Ventilation),
bounded setting adjustments, predictive risk-guided adaptation, and clinician override guardrails.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import numpy as np

@dataclass
class VentilatorSettings:
    fio2: float             # Fraction of inspired oxygen (0.21 - 1.0)
    peep: float             # Positive end-expiratory pressure (cmH2O)
    tidal_volume_ml: float  # Tidal volume (mL)
    respiratory_rate: int   # Breaths per minute (bpm)
    mode: str = "VCV"       # Volume Control Ventilation or PCV

@dataclass
class SafetyConstraints:
    min_fio2: float = 0.21
    max_fio2: float = 1.00
    min_peep: float = 5.0
    max_peep: float = 20.0
    min_vt_per_kg: float = 4.0   # mL/kg PBW
    max_vt_per_kg: float = 8.0   # mL/kg PBW
    target_vt_per_kg: float = 6.0 # mL/kg PBW (Standard Lung-Protective)
    max_plateau_pressure: float = 30.0  # cmH2O (Barotrauma threshold)
    max_driving_pressure: float = 14.0  # cmH2O (Lung injury threshold)
    max_fio2_step: float = 0.10        # Maximum step change per cycle
    max_peep_step: float = 2.0         # Maximum step change per cycle
    max_vt_step_ml: float = 50.0       # Maximum step change per cycle
    target_spo2_min: float = 92.0      # % (Standard target)
    target_spo2_max: float = 96.0      # % (Standard target)
    target_spo2_ards_min: float = 88.0 # % (Permissive hypoxemia in ARDS)
    target_spo2_ards_max: float = 92.0 # %

@dataclass
class AdjustmentRecommendation:
    current_settings: Dict[str, Any]
    proposed_settings: Dict[str, Any]
    setting_deltas: Dict[str, float]
    predicted_risk: float
    urgency_level: str  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    clinical_rationale: List[str]
    safety_checks: Dict[str, bool]
    all_safety_passed: bool
    requires_clinician_approval: bool

class BoundedAdaptiveVentilatorController:
    """
    Adaptive Ventilator Controller adhering to strict Lung-Protective guidelines.
    Combines machine learning risk predictions with clinical protocol bounds.
    """
    def __init__(self, constraints: SafetyConstraints = None):
        self.constraints = constraints or SafetyConstraints()

    def evaluate_and_recommend(
        self,
        current_settings: VentilatorSettings,
        patient_vitals: Dict[str, float],
        pbw_kg: float,
        predicted_risk: float,
        is_ards_or_copd: bool = False
    ) -> AdjustmentRecommendation:
        """
        Determines personalized, bounded adjustments to ventilator settings
        based on current patient physiology, lung mechanics, and AI predicted risk.
        """
        c = self.constraints
        rationale = []
        checks = {}

        # 1. Determine target SpO2 window
        target_spo2_min = c.target_spo2_ards_min if is_ards_or_copd else c.target_spo2_min
        target_spo2_max = c.target_spo2_ards_max if is_ards_or_copd else c.target_spo2_max

        # Current patient observations
        current_spo2 = patient_vitals.get("spo2", 95.0)
        current_rr = patient_vitals.get("respiratory_rate", 18.0)
        plateau_p = patient_vitals.get("plateau_pressure", 22.0)
        driving_p = patient_vitals.get("driving_pressure", plateau_p - current_settings.peep)
        ph = patient_vitals.get("ph", 7.40)
        paco2 = patient_vitals.get("paco2", 40.0)

        # Propose target starting from current
        prop_fio2 = current_settings.fio2
        prop_peep = current_settings.peep
        prop_vt = current_settings.tidal_volume_ml
        prop_rr = current_settings.respiratory_rate

        # Determine Urgency Level from Risk Score
        if predicted_risk < 0.25:
            urgency = "LOW"
        elif predicted_risk < 0.50:
            urgency = "MODERATE"
        elif predicted_risk < 0.75:
            urgency = "HIGH"
        else:
            urgency = "CRITICAL"

        # --- A. OXYGENATION MANAGEMENT (FiO2 & PEEP Titration) ---
        if current_spo2 < target_spo2_min:
            # Hypoxemia detected
            deficit = target_spo2_min - current_spo2
            rationale.append(f"SpO2 ({current_spo2:.1f}%) is below target ({target_spo2_min:.0f}%).")
            
            # Step FiO2 up proportionally, bounded by max_fio2_step
            fio2_step = min(c.max_fio2_step, max(0.05, 0.02 * deficit))
            prop_fio2 = min(c.max_fio2, prop_fio2 + fio2_step)
            rationale.append(f"Titrating FiO2 +{fio2_step:.2f} (Target: {prop_fio2:.2f}).")

            # If hypoxemic and predicted deterioration is high or current FiO2 is already high (>0.50), step up PEEP to recruit alveoli
            if (predicted_risk >= 0.40 or current_settings.fio2 >= 0.50) and prop_peep < c.max_peep:
                peep_step = min(c.max_peep_step, 2.0)
                prop_peep = min(c.max_peep, prop_peep + peep_step)
                rationale.append(f"High risk / refractory hypoxemia: Titrating PEEP +{peep_step:.1f} cmH2O to recruit atelectatic lung units.")

        elif current_spo2 > target_spo2_max and current_settings.fio2 > 0.30:
            # Hyperoxia prevention / De-escalation
            rationale.append(f"SpO2 ({current_spo2:.1f}%) is above target range. Weaning oxygen support to prevent absorption atelectasis and hyperoxic injury.")
            fio2_step = min(c.max_fio2_step, 0.05)
            prop_fio2 = max(c.min_fio2, prop_fio2 - fio2_step)

            # Wean PEEP if FiO2 is modest and PEEP > 5
            if prop_fio2 <= 0.40 and prop_peep > c.min_peep and predicted_risk < 0.30:
                peep_step = min(c.max_peep_step, 1.0)
                prop_peep = max(c.min_peep, prop_peep - peep_step)
                rationale.append(f"Patient stable: Weaning PEEP -{peep_step:.1f} cmH2O.")

        # --- B. LUNG-PROTECTIVE VENTILATION (Driving Pressure & Vt Titration) ---
        ideal_vt = c.target_vt_per_kg * pbw_kg
        min_vt = c.min_vt_per_kg * pbw_kg
        max_vt = c.max_vt_per_kg * pbw_kg

        # Check Driving Pressure
        if driving_p > c.max_driving_pressure:
            rationale.append(f"Excessive driving pressure ({driving_p:.1f} cmH2O > {c.max_driving_pressure} cmH2O). High risk of volutrauma/strain.")
            # Reduce Tidal Volume by bounded step
            vt_reduction = min(c.max_vt_step_ml, max(25.0, (driving_p - c.max_driving_pressure) * 10))
            prop_vt = max(min_vt, prop_vt - vt_reduction)
            rationale.append(f"Reducing tidal volume to {prop_vt:.0f} mL ({prop_vt/pbw_kg:.1f} mL/kg PBW) to lower transpulmonary driving pressure.")

            # Compensate minute ventilation by slightly increasing respiratory rate if pH permits
            if ph < 7.35 and prop_rr < 28:
                prop_rr = min(30, prop_rr + 2)
                rationale.append(f"Compensating minute ventilation: Respiratory rate adjusted to {prop_rr} bpm.")

        elif prop_vt > max_vt:
            prop_vt = max_vt
            rationale.append(f"Capping tidal volume to strict 8.0 mL/kg PBW ({prop_vt:.0f} mL).")

        # --- C. PREDICTIVE DETERIORATION PREEMPTIVE ADJUSTMENT ---
        if predicted_risk >= 0.65 and urgency in ["HIGH", "CRITICAL"]:
            rationale.append(f"AI Model predicts high deterioration risk ({predicted_risk:.1%}) within 4-6h. Enforcing preemptive protective lung stability.")
            if prop_peep < 8.0:
                prop_peep = min(c.max_peep, prop_peep + 2.0)
                rationale.append("Preemptively stabilizing functional residual capacity with PEEP set to 8.0 cmH2O.")

        # --- D. APPLY HARD BOUNDS & STEP LIMITS ---
        prop_fio2 = float(np.clip(prop_fio2, c.min_fio2, c.max_fio2))
        prop_peep = float(np.clip(prop_peep, c.min_peep, c.max_peep))
        prop_vt = float(np.clip(prop_vt, min_vt, max_vt))
        prop_rr = int(np.clip(prop_rr, 10, 32))

        # Enforce rate-of-change limits from current
        prop_fio2 = float(np.clip(prop_fio2, current_settings.fio2 - c.max_fio2_step, current_settings.fio2 + c.max_fio2_step))
        prop_peep = float(np.clip(prop_peep, current_settings.peep - c.max_peep_step, current_settings.peep + c.max_peep_step))
        prop_vt = float(np.clip(prop_vt, current_settings.tidal_volume_ml - c.max_vt_step_ml, current_settings.tidal_volume_ml + c.max_vt_step_ml))
        prop_rr = int(np.clip(prop_rr, current_settings.respiratory_rate - 3, current_settings.respiratory_rate + 3))

        # --- E. SAFETY CONSTRAINT AUDIT ---
        checks["fio2_within_bounds"] = (c.min_fio2 <= prop_fio2 <= c.max_fio2)
        checks["peep_within_bounds"] = (c.min_fio2 <= prop_peep <= c.max_peep)
        checks["vt_per_kg_safe"] = (c.min_vt_per_kg <= (prop_vt / pbw_kg) <= c.max_vt_per_kg)
        checks["max_step_fio2_passed"] = abs(prop_fio2 - current_settings.fio2) <= c.max_fio2_step + 1e-4
        checks["max_step_peep_passed"] = abs(prop_peep - current_settings.peep) <= c.max_peep_step + 1e-4
        checks["plateau_limit_respected"] = (plateau_p <= c.max_plateau_pressure)
        
        all_passed = all(checks.values())

        setting_deltas = {
            "delta_fio2": round(prop_fio2 - current_settings.fio2, 3),
            "delta_peep": round(prop_peep - current_settings.peep, 1),
            "delta_tidal_volume_ml": round(prop_vt - current_settings.tidal_volume_ml, 1),
            "delta_respiratory_rate": int(prop_rr - current_settings.respiratory_rate)
        }

        # Any high-risk or critical adjustment requires clinician review & sign-off
        requires_approval = urgency in ["MODERATE", "HIGH", "CRITICAL"] or any(abs(v) > 0 for v in setting_deltas.values())

        return AdjustmentRecommendation(
            current_settings=asdict(current_settings),
            proposed_settings={
                "fio2": round(prop_fio2, 2),
                "peep": round(prop_peep, 1),
                "tidal_volume_ml": round(prop_vt, 1),
                "respiratory_rate": int(prop_rr),
                "vt_per_kg_pbw": round(prop_vt / pbw_kg, 2),
                "mode": current_settings.mode
            },
            setting_deltas=setting_deltas,
            predicted_risk=round(float(predicted_risk), 3),
            urgency_level=urgency,
            clinical_rationale=rationale if rationale else ["Patient stable within target parameters. Maintain current settings."],
            safety_checks=checks,
            all_safety_passed=all_passed,
            requires_clinician_approval=requires_approval
        )
