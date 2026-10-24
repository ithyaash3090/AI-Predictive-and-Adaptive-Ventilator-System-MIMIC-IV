"""
Physiological Respiratory Simulator for ICU Mechanical Ventilation.
Uses gold-standard Oxygen Content equation (Berggren shunt formula):
Cc'O2, CvO2, and CaO2 = Cc'O2*(1-Qs/Qt) + CvO2*(Qs/Qt).
Accurately models respiratory mechanics and arterial blood gases in ARDS and critical illness.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from src.adaptive.controller import (
    BoundedAdaptiveVentilatorController,
    VentilatorSettings,
    SafetyConstraints
)

@dataclass
class PatientLungState:
    compliance_ml_cmh2o: float   # Respiratory system compliance (mL / cmH2O)
    airway_resistance: float     # Airway resistance (cmH2O / (L/s))
    shunt_fraction_base: float   # Baseline intrapulmonary shunt fraction (0.05 - 0.50)
    dead_space_ml: float         # Anatomical + alveolar dead space (mL)
    hb_g_dl: float = 12.0        # Hemoglobin (g/dL)
    svo2: float = 0.68           # Mixed venous oxygen saturation (fraction)
    vco2_ml_min: float = 220.0   # Metabolic CO2 production (mL/min)
    hco3_meq_l: float = 24.0     # Serum bicarbonate (mEq/L)
    pbw_kg: float = 70.0         # Predicted body weight

class RespiratorySimulator:
    def __init__(self, patient_state: PatientLungState = None):
        self.patient = patient_state or PatientLungState(
            compliance_ml_cmh2o=50.0,
            airway_resistance=8.0,
            shunt_fraction_base=0.15,
            dead_space_ml=150.0,
            pbw_kg=70.0
        )
        self.controller = BoundedAdaptiveVentilatorController()

    def calculate_mechanics_and_gas_exchange(
        self,
        settings: VentilatorSettings,
        patient: PatientLungState
    ) -> Dict[str, float]:
        """
        Computes airway pressures and arterial blood gas values using
        classic Berggren shunt equation and Henderson-Hasselbalch relations.
        """
        # PEEP recruitment effect on compliance
        # PEEP opens collapsed alveoli up to an optimal inflection point
        peep_delta = max(0.0, settings.peep - 5.0)
        peep_recruitment_factor = 1.0 + 0.04 * peep_delta - 0.001 * (peep_delta ** 2)
        effective_compliance = patient.compliance_ml_cmh2o * peep_recruitment_factor
        effective_compliance = max(10.0, min(100.0, effective_compliance))

        # Airway pressures
        driving_pressure = settings.tidal_volume_ml / effective_compliance
        plateau_pressure = settings.peep + driving_pressure
        insp_flow_l_s = 0.8
        peak_insp_pressure = plateau_pressure + (insp_flow_l_s * patient.airway_resistance)

        # Minute and alveolar ventilation
        minute_ventilation_l = (settings.tidal_volume_ml * settings.respiratory_rate) / 1000.0
        alv_tidal_vol = max(40.0, settings.tidal_volume_ml - patient.dead_space_ml)
        alv_ventilation_l = (alv_tidal_vol * settings.respiratory_rate) / 1000.0

        # Carbon Dioxide (PaCO2) & pH
        k_co2 = 0.863
        paco2 = (patient.vco2_ml_min * k_co2) / alv_ventilation_l
        paco2 = float(np.clip(paco2, 22.0, 90.0))

        # Henderson-Hasselbalch equation for arterial pH
        ph = 6.10 + math.log10(patient.hco3_meq_l / (0.0307 * paco2))
        ph = float(np.clip(ph, 6.95, 7.60))

        # Intrapulmonary Shunt (Qs/Qt) with PEEP alveolar recruitment
        effective_shunt = patient.shunt_fraction_base * math.exp(-0.08 * peep_delta)
        effective_shunt = float(np.clip(effective_shunt, 0.05, 0.65))

        # Alveolar PO2 (PAO2)
        p_atm = 760.0
        p_h2o = 47.0
        rq = 0.8
        p_alv_o2 = (p_atm - p_h2o) * settings.fio2 - (paco2 / rq)
        p_alv_o2 = max(35.0, p_alv_o2)

        # Capillary Oxygen Content (Cc'O2 in mL O2 / dL blood)
        # 100% saturation in end-capillaries of ventilated alveoli
        cc_prime_o2 = 1.34 * patient.hb_g_dl * 1.0 + 0.0031 * p_alv_o2

        # Mixed Venous Oxygen Content (Cv_bar_O2)
        # PvO2 approx 36 mmHg
        pv_o2 = 36.0
        cv_o2 = 1.34 * patient.hb_g_dl * patient.svo2 + 0.0031 * pv_o2

        # Arterial Oxygen Content (CaO2) via Berggren shunt:
        # CaO2 = Cc'O2 * (1 - Shunt) + CvO2 * Shunt
        ca_o2 = cc_prime_o2 * (1.0 - effective_shunt) + cv_o2 * effective_shunt

        # Extract Arterial Oxygen Saturation (SaO2 / SpO2) from CaO2:
        # CaO2 approx 1.34 * Hb * SaO2
        max_hb_capacity = 1.34 * patient.hb_g_dl
        sao2_fraction = (ca_o2 - 0.0031 * 80.0) / max_hb_capacity
        sao2_fraction = float(np.clip(sao2_fraction, 0.50, 0.999))
        spo2 = round(sao2_fraction * 100.0, 1)

        # Invert Hill equation to obtain PaO2:
        # SaO2 = PaO2^n / (PaO2^n + P50^n) => PaO2 = P50 * (SaO2 / (1 - SaO2))^(1/n)
        p50 = 26.6
        n = 2.7
        odds = sao2_fraction / (1.0001 - sao2_fraction)
        pao2 = p50 * (odds ** (1.0 / n))
        pao2 = float(np.clip(pao2, 30.0, 450.0))

        # Oxygenation ratios
        sf_ratio = spo2 / max(0.21, settings.fio2)
        pf_ratio = pao2 / max(0.21, settings.fio2)
        rox_index = sf_ratio / max(1.0, float(settings.respiratory_rate))

        return {
            "plateau_pressure": round(plateau_pressure, 1),
            "driving_pressure": round(driving_pressure, 1),
            "peak_insp_pressure": round(peak_insp_pressure, 1),
            "effective_compliance": round(effective_compliance, 1),
            "minute_ventilation": round(minute_ventilation_l, 2),
            "paco2": round(paco2, 1),
            "ph": round(ph, 2),
            "pao2": round(pao2, 1),
            "spo2": round(spo2, 1),
            "effective_shunt": round(effective_shunt, 3),
            "sf_ratio": round(sf_ratio, 1),
            "pf_ratio": round(pf_ratio, 1),
            "rox_index": round(rox_index, 2),
            "vt_per_kg_pbw": round(settings.tidal_volume_ml / patient.pbw_kg, 2)
        }

    def run_comparative_simulation(
        self,
        scenario: str = "acute_lung_injury",
        total_steps: int = 24
    ) -> Dict[str, Any]:
        """
        Runs an hourly 24-hour simulation comparing:
        1. Fixed Standard Ventilator Management
        2. AI-Powered Adaptive & Predictive Ventilation Support System
        """
        base_settings = VentilatorSettings(
            fio2=0.35,
            peep=5.0,
            tidal_volume_ml=480.0,
            respiratory_rate=16,
            mode="VCV"
        )

        patient_fixed = PatientLungState(
            compliance_ml_cmh2o=50.0,
            airway_resistance=8.0,
            shunt_fraction_base=0.12,
            dead_space_ml=150.0,
            pbw_kg=70.0
        )

        patient_adaptive = PatientLungState(
            compliance_ml_cmh2o=50.0,
            airway_resistance=8.0,
            shunt_fraction_base=0.12,
            dead_space_ml=150.0,
            pbw_kg=70.0
        )

        history_fixed = []
        history_adaptive = []

        curr_fixed_settings = VentilatorSettings(
            fio2=base_settings.fio2,
            peep=base_settings.peep,
            tidal_volume_ml=base_settings.tidal_volume_ml,
            respiratory_rate=base_settings.respiratory_rate
        )

        curr_adaptive_settings = VentilatorSettings(
            fio2=base_settings.fio2,
            peep=base_settings.peep,
            tidal_volume_ml=base_settings.tidal_volume_ml,
            respiratory_rate=base_settings.respiratory_rate
        )

        for step in range(total_steps):
            # Induce Acute Deterioration from Step 5 to 16
            if 5 <= step <= 15:
                severity = (step - 4) / 8.0 if step <= 10 else (15 - step) / 5.0
                severity = min(1.0, max(0.0, severity))
                
                # Compliance plummets from 50 down to 22 mL/cmH2O (severe stiffening)
                comp = 50.0 - 28.0 * severity
                # Shunt fraction spikes from 12% to 42% (alveolar collapse)
                shunt = 0.12 + 0.32 * severity
                pred_risk = 0.15 + 0.75 * severity
            elif step > 15:
                recovery = (step - 15) / 9.0
                comp = 22.0 + 22.0 * recovery
                shunt = 0.44 - 0.26 * recovery
                pred_risk = 0.90 - 0.70 * recovery
            else:
                comp = 50.0
                shunt = 0.12
                pred_risk = 0.14

            # Apply identical physiological injury
            patient_fixed.compliance_ml_cmh2o = comp
            patient_fixed.shunt_fraction_base = shunt

            patient_adaptive.compliance_ml_cmh2o = comp
            patient_adaptive.shunt_fraction_base = shunt

            # 1. FIXED VENTILATOR ARM
            fixed_out = self.calculate_mechanics_and_gas_exchange(curr_fixed_settings, patient_fixed)
            fixed_record = {
                "step": step,
                "hour": step,
                "settings": {
                    "fio2": curr_fixed_settings.fio2,
                    "peep": curr_fixed_settings.peep,
                    "tidal_volume_ml": curr_fixed_settings.tidal_volume_ml,
                    "respiratory_rate": curr_fixed_settings.respiratory_rate
                },
                "vitals": fixed_out,
                "predicted_risk": round(pred_risk, 3),
                "lung_injury_alert": fixed_out["driving_pressure"] > 14.0 or fixed_out["spo2"] < 88.0
            }
            history_fixed.append(fixed_record)

            # 2. AI ADAPTIVE VENTILATOR ARM
            adaptive_out = self.calculate_mechanics_and_gas_exchange(curr_adaptive_settings, patient_adaptive)
            rec = self.controller.evaluate_and_recommend(
                current_settings=curr_adaptive_settings,
                patient_vitals=adaptive_out,
                pbw_kg=patient_adaptive.pbw_kg,
                predicted_risk=pred_risk,
                is_ards_or_copd=True
            )

            adaptive_record = {
                "step": step,
                "hour": step,
                "settings": {
                    "fio2": curr_adaptive_settings.fio2,
                    "peep": curr_adaptive_settings.peep,
                    "tidal_volume_ml": curr_adaptive_settings.tidal_volume_ml,
                    "respiratory_rate": curr_adaptive_settings.respiratory_rate
                },
                "vitals": adaptive_out,
                "predicted_risk": round(pred_risk, 3),
                "recommendation": rec,
                "lung_injury_alert": adaptive_out["driving_pressure"] > 14.0 or adaptive_out["spo2"] < 88.0
            }
            history_adaptive.append(adaptive_record)

            # Apply adaptive controller recommendations to next step
            curr_adaptive_settings = VentilatorSettings(
                fio2=rec.proposed_settings["fio2"],
                peep=rec.proposed_settings["peep"],
                tidal_volume_ml=rec.proposed_settings["tidal_volume_ml"],
                respiratory_rate=rec.proposed_settings["respiratory_rate"]
            )

        # Aggregate comparative summary statistics
        fixed_spo2 = [r["vitals"]["spo2"] for r in history_fixed]
        adapt_spo2 = [r["vitals"]["spo2"] for r in history_adaptive]
        fixed_dp = [r["vitals"]["driving_pressure"] for r in history_fixed]
        adapt_dp = [r["vitals"]["driving_pressure"] for r in history_adaptive]
        fixed_pplat = [r["vitals"]["plateau_pressure"] for r in history_fixed]
        adapt_pplat = [r["vitals"]["plateau_pressure"] for r in history_adaptive]

        summary = {
            "fixed_strategy": {
                "min_spo2": min(fixed_spo2),
                "mean_spo2": round(float(np.mean(fixed_spo2)), 1),
                "max_driving_pressure": max(fixed_dp),
                "mean_driving_pressure": round(float(np.mean(fixed_dp)), 1),
                "max_plateau_pressure": max(fixed_pplat),
                "hours_in_hypoxemia": sum(1 for s in fixed_spo2 if s < 88.0),
                "hours_in_high_driving_pressure": sum(1 for dp in fixed_dp if dp > 14.0)
            },
            "adaptive_strategy": {
                "min_spo2": min(adapt_spo2),
                "mean_spo2": round(float(np.mean(adapt_spo2)), 1),
                "max_driving_pressure": max(adapt_dp),
                "mean_driving_pressure": round(float(np.mean(adapt_dp)), 1),
                "max_plateau_pressure": max(adapt_pplat),
                "hours_in_hypoxemia": sum(1 for s in adapt_spo2 if s < 88.0),
                "hours_in_high_driving_pressure": sum(1 for dp in adapt_dp if dp > 14.0)
            }
        }

        return {
            "scenario": scenario,
            "total_steps": total_steps,
            "summary": summary,
            "fixed_history": history_fixed,
            "adaptive_history": history_adaptive
        }
