# AI-Powered Adaptive and Predictive Ventilation Support System for ICU Patients

An end-to-end, clinically grounded artificial intelligence system that combines **early predictive deterioration modeling** on MIMIC-IV clinical time-series with a **bounded adaptive ventilation controller** and an interactive, real-time clinical monitoring station.

--- 

## 🌟 Key Features:

1. **Retrospective MIMIC-IV Clinical Pipeline (`src/data/extractor.py`)**
   - Ingests ICU stays, patient demographics, ICD diagnoses/comorbidities, flow-sheet ventilator measurements (`chartevents`), and blood gas laboratory panels (`labevents`).
   - Aligns physiological measurements onto continuous hourly grids (10,099 hourly records across 139 ICU stays and 100 unique patients).
   - Extracts non-invasive and invasive clinical indicators: $SpO_2 / FiO_2$ (S/F ratio), $PaO_2 / FiO_2$ (P/F ratio), ROX Index, Driving Pressure ($\Delta P = P_{\text{plat}} - \text{PEEP}$), and PBW-normalized tidal volumes ($V_t\text{ / kg PBW}$).

2. **Comparative Machine Learning Suite (`src/models/trainer.py`)**
   - Gold-standard **GroupKFold (k=5)** cross-validation strictly grouped by patient ID (`subject_id`) to avoid cross-admission contamination.
   - Evaluates:
     - **Logistic Regression (L2)** (Clinical Baseline)
     - **Random Forest Classifier**
     - **LightGBM Classifier**
     - **XGBoost Classifier**
   - **Performance Results**:
     - **XGBoost:** AUROC **0.9376 &plusmn; 0.0227**, AUPRC **0.8214 &plusmn; 0.0564**, Specificity **96.47%**, Sensitivity **76.65%**, Brier score **0.0465**.
     - **Average Warning Lead Time:** **3.83 Hours** prior to clinical deterioration onset.
     - **Top Clinical Features:** S/F ratio ($0.3786$), P/F ratio ($0.1010$), rolling mean $FiO_2$ ($0.0480$), driving pressure ($0.0336$), arterial lactate ($0.0243$), and pre-existing respiratory failure ($0.0224$).

3. **Bounded Adaptive Ventilation Controller (`src/adaptive/controller.py`)**
   - Multi-objective optimization within strict **ARDSNet Lung-Protective Guidelines**:
     - Tidal Volume: strictly bounded between $4.0$ and $8.0\text{ mL/kg PBW}$ (target: $6.0\text{ mL/kg}$).
     - Barotrauma Ceiling: Plateau Pressure strictly $\le 30\text{ cmH}_2\text{O}$.
     - Lung Strain Bound: Driving Pressure strictly $\le 14\text{ cmH}_2\text{O}$.
     - Rate-of-Change Limiters: $|\Delta FiO_2| \le 0.10$, $|\Delta \text{PEEP}| \le 2.0\text{ cmH}_2\text{O}$ per cycle.
     - Tiered Clinician-in-the-Loop Supervision & affirmative approval states.

4. **Closed-Loop Physiological Lung Simulator (`src/adaptive/simulator.py`)**
   - 2-compartment mathematical mechanics model ($C_{\text{rs}}, R_{\text{aw}}, \Delta P$) coupled with the classic **Berggren shunt equation** ($CaO_2 = Cc'O_2 \cdot (1 - Q_s/Q_t) + C\bar{v}O_2 \cdot (Q_s/Q_t)$).
   - Compares **Fixed Standard Settings** vs **AI Adaptive Strategy** during acute lung injury.
   - Results: The AI-adaptive controller **eliminates all hypoxemic hours** ($0\text{ vs }1\text{ hr}$) and **reduces high driving pressure exposure by 50%** ($4\text{ vs }8\text{ hrs}$) without overshoot.

5. **Research-Grade Interactive Web Dashboard (`src/web/`)**
   - Modern, responsive dark-mode ICU station interface built with vanilla CSS glassmorphism and Chart.js.
   - Includes real-time patient telemetry, risk gauges, clinical driver breakdowns, adaptive recommendations, 7-point safety checklists, benchmark ROC/PR curves, and live simulation sandboxes.

6. **Full Academic Research Paper (`research_paper.md`)**
   - Publication-ready manuscript documenting clinical problem, prior art, methodology, experimental results, ethical governance, and translational roadmap.

---

## 📂 Project Structure

```
├── mimic-iv-demo-dataset/        # Raw MIMIC-IV Clinical Database Demo v2.2
│   └── mimic-iv-clinical-database-demo-2.2/
│       ├── hosp/                 # Hospital tables (patients, admissions, diagnoses, labs)
│       └── icu/                  # ICU tables (icustays, chartevents, procedureevents, d_items)
├── src/
│   ├── data/
│   │   └── extractor.py          # Data ingestion, hourly alignment, and deterioration labeling
│   ├── models/
│   │   └── trainer.py            # Feature engineering, GroupKFold benchmark, feature importances
│   ├── adaptive/
│   │   ├── controller.py         # ARDSNet bounded adaptive controller & 7-point safety audit
│   │   └── simulator.py          # 2-compartment lung simulator with Berggren shunt equation
│   └── web/
│       ├── app.py                # FastAPI REST API backend
│       ├── templates/
│       │   └── dashboard.html    # Clinical monitoring dashboard HTML
│       └── static/
│           ├── styles.css        # Glassmorphic dark-mode CSS design system
│           └── dashboard.js      # Frontend telemetry and Chart.js rendering
├── scripts/
│   ├── build_dataset.py          # Generates processed MIMIC-IV time-series
│   ├── train_models.py           # Runs GroupKFold cross-validation benchmark
│   └── run_simulation.py         # Executes physiological lung simulation
├── outputs/
│   ├── processed_mimic_cohort.csv # Extracted hourly cohort (10,099 records)
│   ├── processed_cohort_with_predictions.csv
│   ├── benchmark_results.json    # AUROC, AUPRC, ROC/PR curves, feature rankings
│   ├── simulation_results.json   # Simulation telemetry trajectories
│   └── models/                   # Saved XGBoost, LightGBM, RF, scaler, and imputer artifacts
├── research_paper.md             # Complete academic research paper ready for submission
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.10+
- Installed packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `fastapi`, `uvicorn`, `jinja2`

### 2. Extract and Preprocess MIMIC-IV Cohort
```bash
python scripts/build_dataset.py
```
*Generates `outputs/processed_mimic_cohort.csv` with 10,099 hourly records and derived clinical metrics.*

### 3. Train and Benchmark Predictive Models
```bash
python scripts/train_models.py
```
*Executes patient-level GroupKFold cross-validation across Logistic Regression, Random Forest, LightGBM, and XGBoost. Saves model artifacts and metrics to `outputs/`.*

### 4. Run Closed-Loop Physiological Lung Simulation
```bash
python scripts/run_simulation.py
```
*Executes 24-hour acute lung injury simulation comparing Fixed vs AI Adaptive management.*

### 5. Launch the Interactive Clinical Dashboard
```bash
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```
Open your browser to: **`http://127.0.0.1:8000`**

---

## 📊 Summary of Benchmark Results

| Model Architecture | AUROC | AUPRC | Sensitivity | Specificity | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.8889 \pm 0.0263$ | $0.7172 \pm 0.0350$ | $79.89\%$ | $81.76\%$ | $0.5642$ | $0.1059$ |
| **Random Forest** | $0.9397 \pm 0.0229$ | $0.8105 \pm 0.0540$ | $77.64\%$ | $96.31\%$ | $0.7661$ | $0.0530$ |
| **LightGBM** | $0.9345 \pm 0.0248$ | $0.8154 \pm 0.0647$ | $76.89\%$ | **$96.83\%$** | **$0.7765$** | **$0.0452$** |
| **XGBoost** | **$0.9376 \pm 0.0227$** | **$0.8214 \pm 0.0564$** | $76.65\%$ | $96.47\%$ | $0.7680$ | $0.0465$ |

**Average Early Warning Lead Time:** **3.83 Hours** prior to acute respiratory deterioration.

---

## 🛡️ Clinical Safety & Governance Principles

1. **Strict ARDSNet Envelopes:** No recommendation violates the $4–8\text{ mL/kg PBW}$ tidal volume bound or allows plateau pressure to exceed $30\text{ cmH}_2\text{O}$.
2. **Anti-Fatigue Specificity:** By achieving $>96.4\%$ specificity, the system suppresses false alarms that plague conventional monitors.
3. **Clinician In The Loop:** Autonomous mode is isolated to research simulation; clinical station mode mandates clinician confirmation for significant setting migrations and provides instantaneous manual override.
