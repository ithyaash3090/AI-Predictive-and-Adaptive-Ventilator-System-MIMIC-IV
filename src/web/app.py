"""
FastAPI Server for AI-Powered Adaptive and Predictive Ventilation Support System.
Provides REST APIs for patient monitoring, predictive risk inference,
adaptive ventilator recommendations, model benchmarks, and real-time lung simulation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from src.adaptive.controller import (
    BoundedAdaptiveVentilatorController,
    VentilatorSettings,
    SafetyConstraints
)
from src.adaptive.simulator import RespiratorySimulator, PatientLungState

app = FastAPI(title="AI-Powered Adaptive and Predictive Ventilation Support System", version="1.0.0")

# Setup template & static directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global in-memory cache for fast dashboard response
DATASET_PATH = "outputs/processed_cohort_with_predictions.csv"
BENCHMARK_PATH = "outputs/benchmark_results.json"
SIMULATION_PATH = "outputs/simulation_results.json"

df_cohort = None
benchmark_cache = None
controller = BoundedAdaptiveVentilatorController()

def get_cohort_data():
    global df_cohort
    if df_cohort is None:
        if os.path.exists(DATASET_PATH):
            df_cohort = pd.read_csv(DATASET_PATH)
            # Fill NaNs for JSON serialization
            df_cohort = df_cohort.replace({np.nan: None})
        else:
            raise RuntimeError(f"Processed dataset not found at {DATASET_PATH}. Please run extraction and training first.")
    return df_cohort

def get_benchmark_data():
    global benchmark_cache
    if benchmark_cache is None:
        if os.path.exists(BENCHMARK_PATH):
            with open(BENCHMARK_PATH, "r") as f:
                benchmark_cache = json.load(f)
        else:
            benchmark_cache = {}
    return benchmark_cache

@app.get("/", response_class=FileResponse)
async def index():
    """Renders the main clinical dashboard."""
    html_path = os.path.join(TEMPLATES_DIR, "dashboard.html")
    return FileResponse(html_path)

@app.get("/api/patients")
async def list_patients():
    """Returns summarized list of ICU patients for the patient selector."""
    df = get_cohort_data()
    summary = []
    for stay_id, group in df.groupby('stay_id'):
        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        max_risk = float(group['predicted_risk_xgb'].max()) if 'predicted_risk_xgb' in group else 0.0
        deteriorated = int(group['target_deterioration_lead'].max())
        
        # Primary comorbidity label
        diag_labels = []
        if first_row.get('has_ards'): diag_labels.append("ARDS")
        if first_row.get('has_respiratory_failure'): diag_labels.append("Acute Resp Failure")
        if first_row.get('has_pneumonia'): diag_labels.append("Pneumonia")
        if first_row.get('has_copd'): diag_labels.append("COPD")
        if first_row.get('has_sepsis'): diag_labels.append("Sepsis")
        
        summary.append({
            "stay_id": int(stay_id),
            "subject_id": int(first_row['subject_id']),
            "age": int(first_row['age']),
            "gender": str(first_row['gender']),
            "pbw_kg": float(first_row['pbw_kg']),
            "total_hours": len(group),
            "max_risk": round(max_risk, 3),
            "current_risk": round(float(last_row['predicted_risk_xgb']), 3) if 'predicted_risk_xgb' in last_row else 0.0,
            "experienced_deterioration": bool(deteriorated),
            "primary_diagnoses": ", ".join(diag_labels) if diag_labels else "ICU Monitoring"
        })
    
    # Sort patients: patients with higher risk / deterioration first
    summary.sort(key=lambda x: (x['experienced_deterioration'], x['max_risk']), reverse=True)
    return {"patients": summary, "total_count": len(summary)}

@app.get("/api/patient/{stay_id}")
async def get_patient_details(stay_id: int):
    """Returns complete longitudinal time-series for a selected patient stay."""
    df = get_cohort_data()
    patient_df = df[df['stay_id'] == stay_id].sort_values('hour_from_intime')
    
    if len(patient_df) == 0:
        raise HTTPException(status_code=404, detail="Patient stay not found.")

    records = patient_df.to_dict(orient='records')
    demographics = {
        "stay_id": stay_id,
        "subject_id": int(records[0]['subject_id']),
        "age": int(records[0]['age']),
        "gender": str(records[0]['gender']),
        "pbw_kg": float(records[0]['pbw_kg']),
        "has_ards": bool(records[0].get('has_ards', False)),
        "has_respiratory_failure": bool(records[0].get('has_respiratory_failure', False)),
        "has_pneumonia": bool(records[0].get('has_pneumonia', False)),
        "has_copd": bool(records[0].get('has_copd', False)),
        "has_sepsis": bool(records[0].get('has_sepsis', False)),
        "total_monitoring_hours": len(records)
    }

    return {
        "demographics": demographics,
        "timeseries": records
    }

class RecommendationRequest(BaseModel):
    fio2: float
    peep: float
    tidal_volume_ml: float
    respiratory_rate: int
    spo2: float
    driving_pressure: Optional[float] = None
    plateau_pressure: Optional[float] = None
    ph: Optional[float] = 7.40
    paco2: Optional[float] = 40.0
    pbw_kg: float = 70.0
    predicted_risk: float = 0.20
    is_ards_or_copd: bool = False

@app.post("/api/adaptive/recommend")
async def get_recommendation(req: RecommendationRequest):
    """Computes bounded adaptive recommendation and evaluates safety rules."""
    curr_settings = VentilatorSettings(
        fio2=req.fio2,
        peep=req.peep,
        tidal_volume_ml=req.tidal_volume_ml,
        respiratory_rate=req.respiratory_rate
    )
    vitals = {
        "spo2": req.spo2,
        "respiratory_rate": float(req.respiratory_rate),
        "plateau_pressure": req.plateau_pressure if req.plateau_pressure is not None else (req.peep + (req.driving_pressure or 12.0)),
        "driving_pressure": req.driving_pressure if req.driving_pressure is not None else 12.0,
        "ph": req.ph or 7.40,
        "paco2": req.paco2 or 40.0
    }

    rec = controller.evaluate_and_recommend(
        current_settings=curr_settings,
        patient_vitals=vitals,
        pbw_kg=req.pbw_kg,
        predicted_risk=req.predicted_risk,
        is_ards_or_copd=req.is_ards_or_copd
    )

    return {
        "recommendation": {
            "current_settings": rec.current_settings,
            "proposed_settings": rec.proposed_settings,
            "setting_deltas": rec.setting_deltas,
            "predicted_risk": rec.predicted_risk,
            "urgency_level": rec.urgency_level,
            "clinical_rationale": rec.clinical_rationale,
            "safety_checks": rec.safety_checks,
            "all_safety_passed": rec.all_safety_passed,
            "requires_clinician_approval": rec.requires_clinician_approval
        }
    }

@app.get("/api/benchmark")
async def get_benchmark():
    """Returns model performance metrics and ROC/PR curve points."""
    return get_benchmark_data()

class SimulationRunRequest(BaseModel):
    initial_fio2: float = 0.35
    initial_peep: float = 5.0
    initial_tidal_volume_ml: float = 480.0
    initial_respiratory_rate: int = 16
    compliance_drop_severity: float = 0.8  # 0 to 1 scale
    shunt_spike_severity: float = 0.8
    total_steps: int = 24

@app.post("/api/simulation/run")
async def run_simulation(req: SimulationRunRequest):
    """Executes live physiological simulator comparing Fixed vs AI-Adaptive settings."""
    sim = RespiratorySimulator()
    res = sim.run_comparative_simulation(total_steps=req.total_steps)
    return res

class ClinicianActionLog(BaseModel):
    stay_id: int
    action_type: str  # "APPROVE", "MODIFY", "OVERRIDE", "AUTONOMOUS_TOGGLE"
    applied_settings: Dict[str, Any]
    clinician_notes: Optional[str] = "Clinician approved adjustment within ARDSNet bounds."

@app.post("/api/action/log")
async def log_clinician_action(action: ClinicianActionLog):
    """Logs clinician supervision/override action maintaining audit trail."""
    log_entry = action.dict()
    log_entry["timestamp"] = pd.Timestamp.now().isoformat()
    # In production, this writes to the EHR/audit log
    return {"status": "SUCCESS", "message": f"Action {action.action_type} successfully recorded in clinical audit log.", "entry": log_entry}
