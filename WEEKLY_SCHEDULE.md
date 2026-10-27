# 🗓️ Master Weekly Task Schedule (Weeks 0 – 12)

> [!CAUTION]
> 🔒 **ACCESS RESTRICTED: READ-ONLY FOR TEAMMATES, WRITE-ACCESS LEAD-ONLY**  
> **Teammates (Manju, Thiyaa, Mithun) can freely read this file to view schedules and roadmap, but only `ithyaash3090` has the access to edit or modify `WEEKLY_SCHEDULE.md`.**  
> Teammates must NOT edit, stage, or modify this file in their branches or Pull Requests. The schedule is managed and updated exclusively by `@ithyaash3090` after evaluating PRs, running CI checks, and verifying merge status. Any Pull Request attempting to modify this file will be automatically rejected.

This schedule defines the roadmap for the **AI-Powered Adaptive and Predictive Ventilation Support System**.
- **Pacing**: 3 tasks per week per member (assigned every 2 days: Days 1–2, Days 3–4, Days 5–6; Day 7 for review and merging).
- **Rule**: All work must be completed on feature branches (`task/<task-name>`) and submitted via Pull Request to `main`.

---

## 🔹 Phase 0: Team Onboarding & Git Workflow Tutorial (Week 0)

*The objective of Week 0 is for the teammates (Manju, Thiyaa, Mithun) to learn the daily task assignment, branch creation, commit, PR review, and merge conflict resolution with beginner-friendly starter tasks, practicing the rule of zero direct pushes to `main`.*

> 📌 **NOTE**: Week 0 will **officially begin only when the lead announces it**. Any individual warm-up tasks assigned before that are independent practice exercises.

> [!IMPORTANT]
> 🛑 **MANDATORY POLICY FOR WEEK 0 (HANDS-ON MANUAL GIT MASTERY)**:  
> - **The Goal**: Build genuine personal confidence and terminal muscle memory with Git: creating a branch (`git checkout -b ...`), committing, pushing to remote (`git push -u origin ...`), opening a PR, understanding merge conflicts, and syncing `main`.  
> - **AI Usage in Week 0**: You may ask AI questions to learn or clarify git concepts, but **AI MUST NOT write or complete your Week 0 tasks for you**. Every terminal command and script must be executed by you manually.  
> - *(From Week 1 onwards, using AI assistance across the project is completely welcomed!)*

### 📅 Week 0: Starter Tutorial & Git Practice
#### Slot 1 (Days 1–2) — Starter Tutorial Tasks
- **Manju**: `[Task 0.1]` Starter Task: Repository setup, dependencies test & dataset path verification script (`scripts/verify_data_paths.py`) *(Completed: GitHub Issue #3 / PR #6)*.
- **Thiyaa**: `[Task 0.2]` Starter Task: Web server local boot & static frontend asset check script (`scripts/verify_web_assets.py`) *(Completed: GitHub Issue #4 / PR #7)*.
- **Mithun**: `[Task 0.3]` Starter Task: Environment verification & dependency sanity check script (`scripts/verify_setup.py`) *(Completed: GitHub Issue #1 / PR #2)*.

#### Slot 2 (Days 3–4) — First PR Creation & Verification
- **Manju**: `[Task 0.4]` Push branch `task/verify-data-paths` and open Pull Request referencing Issue #3; verify no merge conflicts *(Merged into main via PR #6)*.
- **Thiyaa**: `[Task 0.5]` Push branch `task/verify-web-assets` and open Pull Request referencing Issue #4; verify no merge conflicts *(Merged into main via PR #7)*.
- **Mithun**: `[Task 0.6]` Push branch `task/verify-setup` and open Pull Request referencing Issue #1; verify no merge conflicts *(Merged into main via PR #2)*.

#### Slot 3 (Days 5–6) — Local Sync & Script Run Verification
- **Manju**: `[Task 0.7]` Sync local `main` branch (`git checkout main && git pull origin main`) and test running `python scripts/verify_data_paths.py` *(Active: GitHub Issue #8 — Deadline: 28 Oct 11:59 PM IST)*.
- **Thiyaa**: `[Task 0.8]` Sync local `main` branch (`git checkout main && git pull origin main`) and test running `python scripts/verify_web_assets.py` *(Active: GitHub Issue #9 — Deadline: 28 Oct 11:59 PM IST)*.
- **Mithun**: `[Task 0.9]` Sync local `main` branch (`git checkout main && git pull origin main`) and test running `python scripts/verify_setup.py` & unified multi-script verification (`scripts/verify_all.py`) *(Completed: GitHub Issue #10 / PR #11)*.

#### Day 7 (Day 7) — Week 0 Final Integration, Team Sync & Milestone Wrap-Up
- **Manju**: `[Task 0.10]` Full verification suite execution (`python scripts/verify_all.py`) & final sync confirmation *(Active: GitHub Issue #16 — Deadline: 28 Oct 11:59 PM IST)*.
- **Thiyaa**: `[Task 0.11]` Full verification suite execution (`python scripts/verify_all.py`) & final sync confirmation *(Active: GitHub Issue #17 — Deadline: 28 Oct 11:59 PM IST)*.
- **Mithun**: `[Task 0.12]` Branch cleanup, full suite verification & Week 0 milestone wrap-up *(Active: GitHub Issue #18 — Deadline: 28 Oct 11:59 PM IST)*.

---

## 🔹 Phase 1: Foundation, Data Ingestion & Setup (Weeks 1 – 3)

### 📅 Week 1: Environment, Ingestion & Baseline Sanity
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 1.1]` Create patient demographic summary script (`scripts/eda_demographics.py`).
- **Thiyaa**: `[Task 1.2]` Verify local FastAPI server & static asset loading (`scripts/verify_server.py`).
- **Mithun**: `[Task 1.3]` Create environment verification and dependency sanity script (`scripts/verify_setup.py`).
- **Ithyaash**: `[Task 1.4]` Define core model training abstractions and pipeline interfaces in `src/models/trainer.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 1.5]` Generate clinical vital signs distribution histograms (`scripts/plot_vital_distributions.py`).
- **Thiyaa**: `[Task 1.6]` Responsive layout audit and UI component review (`docs/ui_layout_audit.md`).
- **Mithun**: `[Task 1.7]` Create dataset table inspector script (`scripts/inspect_tables.py`).
- **Ithyaash**: `[Task 1.8]` Audit clinical deterioration threshold labeling logic in `src/data/extractor.py`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 1.9]` Clinical comorbidity frequency analysis (`scripts/eda_comorbidities.py`).
- **Thiyaa**: `[Task 1.10]` Design clinical telemetry status badge tokens in `src/web/static/styles.css`.
- **Mithun**: `[Task 1.11]` Implement missing vital signs audit script (`scripts/audit_missing_values.py`).
- **Ithyaash**: `[Task 1.12]` Implement patient-level cross-validation splitter blueprint (`scripts/test_splitter.py`).

---

### 📅 Week 2: Data Cleaning, Missing Audits & Interface Standards
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 2.1]` Compute correlation matrix across baseline clinical vitals (`scripts/plot_correlations.py`).
- **Thiyaa**: `[Task 2.2]` Implement accessible dark-mode color palette tokens in `src/web/static/styles.css`.
- **Mithun**: `[Task 2.3]` Build laboratory blood gas panel validator for pH, PaCO2, and PaO2 (`scripts/validate_labs.py`).
- **Ithyaash**: `[Task 2.4]` Establish data preprocessing schema and validation contracts in `src/data/schemas.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 2.5]` Generate boxplots for physiological parameters across ICU stay duration (`scripts/plot_vitals_boxplots.py`).
- **Thiyaa**: `[Task 2.6]` Design modular patient telemetry card layout in `src/web/templates/dashboard.html`.
- **Mithun**: `[Task 2.7]` Build flow-sheet ventilator measurements cleaner (`scripts/clean_ventilator_data.py`).
- **Ithyaash**: `[Task 2.8]` Refine hourly grid alignment algorithm in `src/data/extractor.py`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 2.9]` Create cohort summary table generator formatted in markdown (`scripts/generate_cohort_table.py`).
- **Thiyaa**: `[Task 2.10]` Add interactive hover animations and micro-transitions for UI cards (`styles.css`).
- **Mithun**: `[Task 2.11]` Build missing value imputation benchmark comparing mean vs median vs forward-fill (`scripts/impute_benchmark.py`).
- **Ithyaash**: `[Task 2.12]` Implement patient exclusion and quality control filters in `src/data/extractor.py`.

---

### 📅 Week 3: Cohort Finalization & Pipeline Hardening
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 3.1]` Generate paired scatter plots for PaO2/FiO2 vs SpO2/FiO2 indices (`scripts/plot_oxygenation_ratios.py`).
- **Thiyaa**: `[Task 3.2]` Setup client-side state container for real-time telemetry updates in `src/web/static/dashboard.js`.
- **Mithun**: `[Task 3.3]` Script automated cohort consistency check between raw tables and processed CSV (`scripts/verify_cohort_integrity.py`).
- **Ithyaash**: `[Task 3.4]` Implement end-to-end dataset build orchestrator in `scripts/build_dataset.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 3.5]` Calculate deterioration event frequency by hour of ICU admission (`scripts/plot_deterioration_timeline.py`).
- **Thiyaa**: `[Task 3.6]` Build telemetry threshold warning toast notification component (`dashboard.js`, `styles.css`).
- **Mithun**: `[Task 3.7]` Validate driving pressure calculations ($P_{\text{plat}} - \text{PEEP}$) across all ventilator records (`scripts/validate_driving_pressure.py`).
- **Ithyaash**: `[Task 3.8]` Build reproducible cohort caching and versioning manager in `src/data/dataset_manager.py`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 3.9]` Document complete clinical data dictionary in `docs/data_dictionary.md`.
- **Thiyaa**: `[Task 3.10]` Conduct cross-browser UI verification (Chrome, Edge, Firefox).
- **Mithun**: `[Task 3.11]` Create CLI argument parser for dataset extraction parameters (`scripts/build_dataset.py`).
- **Ithyaash**: `[Task 3.12]` Phase 1 final review, integration test execution, and tag `v0.1-phase1-complete`.

---

## 🔹 Phase 2: Feature Engineering & Machine Learning (Weeks 4 – 7)

### 📅 Week 4: Temporal Feature Engineering
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 4.1]` Plot trend lines of vital signs for deteriorating vs non-deteriorating patient stays (`scripts/plot_stay_trajectories.py`).
- **Thiyaa**: `[Task 4.2]` Create telemetry graph placeholder containers using Chart.js in `dashboard.html`.
- **Mithun**: `[Task 4.3]` Implement 3-hour rolling mean and standard deviation feature generator (`src/features/rolling.py`).
- **Ithyaash**: `[Task 4.4]` Design feature transformation pipeline architecture in `src/models/trainer.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 4.5]` Analyze distribution shifts between raw vitals and rolling aggregates (`scripts/plot_feature_distributions.py`).
- **Thiyaa**: `[Task 4.6]` Implement SVG risk gauge component in `src/web/static/dashboard.js`.
- **Mithun**: `[Task 4.7]` Implement 3-hour delta (slope/rate of change) features for SpO2 and FiO2 (`src/features/deltas.py`).
- **Ithyaash**: `[Task 4.8]` Implement composite temporal feature assembler in `src/models/trainer.py`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 4.9]` Evaluate collinearity among rolling features and calculate Variance Inflation Factors (`scripts/vif_analysis.py`).
- **Thiyaa**: `[Task 4.10]` Add real-time visual pulsing indicator for active ICU bed monitoring (`styles.css`).
- **Mithun**: `[Task 4.11]` Integrate ROX Index and driving pressure delta calculations (`src/features/deltas.py`).
- **Ithyaash**: `[Task 4.12]` Implement feature matrix validator and missing rate guards in `src/models/trainer.py`.

---

### 📅 Week 5: Patient-Level Validation & Baseline Modeling
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 5.1]` Build ROC curve and Precision-Recall curve generator module (`scripts/evaluation_metrics.py`).
- **Thiyaa**: `[Task 5.2]` Design bedside patient profile header with PBW and admission details in `dashboard.html`.
- **Mithun**: `[Task 5.3]` Build feature scaler and median imputer wrapper (`src/features/preprocessors.py`).
- **Ithyaash**: `[Task 5.4]` Implement strict patient-level GroupKFold (k=5) evaluation harness in `src/models/trainer.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 5.5]` Build confusion matrix plotting script with customizable thresholds (`scripts/plot_confusion_matrix.py`).
- **Thiyaa**: `[Task 5.6]` Integrate Chart.js time-series configuration for multi-parameter vitals rendering (`dashboard.js`).
- **Mithun**: `[Task 5.7]` Script class imbalance distribution audit across validation folds (`scripts/audit_fold_balance.py`).
- **Ithyaash**: `[Task 5.8]` Implement Logistic Regression (L2) clinical baseline model in `src/models/trainer.py`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 5.9]` Create baseline model performance markdown summary table (`outputs/baseline_report.md`).
- **Thiyaa**: `[Task 5.10]` Style risk score progress bar and warning thresholds in `styles.css`.
- **Mithun**: `[Task 5.11]` Script out-of-fold prediction aggregator and persistence helper (`scripts/save_predictions.py`).
- **Ithyaash**: `[Task 5.12]` Implement Random Forest ensemble classifier with balanced subsampling in `src/models/trainer.py`.

---

### 📅 Week 6: Gradient Boosting (LightGBM & XGBoost)
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 6.1]` Plot comparative ROC curves: Logistic Regression vs Random Forest (`scripts/plot_roc_comparison.py`).
- **Thiyaa**: `[Task 6.2]` Build clinical drivers breakdown panel in `dashboard.html` for feature contribution display.
- **Mithun**: `[Task 6.3]` Prepare hyperparameter grid search configuration for tree-based models (`configs/model_params.json`).
- **Ithyaash**: `[Task 6.4]` Implement LightGBM classifier with early stopping and GroupKFold evaluation in `src/models/trainer.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 6.5]` Calculate Brier calibration scores and generate reliability curves (`scripts/plot_calibration_curve.py`).
- **Thiyaa**: `[Task 6.6]` Build dynamic alert sound toggle and visual alarm banners in `dashboard.js`.
- **Mithun**: `[Task 6.7]` Script cross-validation fold score logger across AUROC, AUPRC, and F1 (`scripts/log_cv_scores.py`).
- **Ithyaash**: `[Task 6.8]` Implement XGBoost classifier with scale_pos_weight optimization in `src/models/trainer.py`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 6.9]` Extract and plot Gini and gain-based feature importance rankings (`scripts/plot_feature_importance.py`).
- **Thiyaa**: `[Task 6.10]` Style comparative model selector dropdown in the dashboard navigation bar (`styles.css`).
- **Mithun**: `[Task 6.11]` Build model serialization and metadata registry (`src/models/registry.py`).
- **Ithyaash**: `[Task 6.12]` Implement threshold calibration for target specificity (>95%) in `src/models/trainer.py`.

---

### 📅 Week 7: Model Benchmarking & Lead-Time Analysis
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 7.1]` Calculate early warning lead time distribution across true positive alerts (`scripts/lead_time_analysis.py`).
- **Thiyaa**: `[Task 7.2]` Build early warning lead-time badge and countdown clock component in `dashboard.js`.
- **Mithun**: `[Task 7.3]` Script full cohort prediction generation across all 10,099 rows (`scripts/generate_predictions.py`).
- **Ithyaash**: `[Task 7.4]` Implement comparative multi-model benchmark executor in `scripts/train_models.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 7.5]` Generate publication-grade benchmark table across all 4 models (`outputs/final_benchmark.md`).
- **Thiyaa**: `[Task 7.6]` Wire dynamic model metrics into the dashboard benchmark tab (`dashboard.js`).
- **Mithun**: `[Task 7.7]` Verify prediction output schema and sanity-check probability ranges (`scripts/verify_predictions.py`).
- **Ithyaash**: `[Task 7.8]` Implement automated model saving and artifact versioning in `outputs/models/`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 7.9]` Generate SHAP summary plots for top clinical predictors (`scripts/plot_shap_summary.py`).
- **Thiyaa**: `[Task 7.10]` Add tooltips explaining clinical indicators (ROX, Driving Pressure, S/F ratio) in the dashboard.
- **Mithun**: `[Task 7.11]` Benchmark model inference latency per sample (`scripts/benchmark_latency.py`).
- **Ithyaash**: `[Task 7.12]` Phase 2 milestone review, model reproducibility verification, and tag `v0.2-phase2-complete`.

---

## 🔹 Phase 3: Adaptive Control & Physiological Simulation (Weeks 8 – 9)

### 📅 Week 8: ARDSNet Bounded Adaptive Controller
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 8.1]` Generate visual ARDSNet PEEP/FiO2 titration grid matrix (`scripts/plot_ardsnet_table.py`).
- **Thiyaa**: `[Task 8.2]` Build 7-point clinical safety checklist UI card in `dashboard.html`.
- **Mithun**: `[Task 8.3]` Build test suite for PBW tidal volume bounds: $4.0 \le V_t \le 8.0\text{ mL/kg}$ (`tests/test_tv_bounds.py`).
- **Ithyaash**: `[Task 8.4]` Implement bounded adaptive ventilation controller architecture in `src/adaptive/controller.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 8.5]` Plot step-change trajectory constraints for PEEP and FiO2 (`scripts/plot_controller_steps.py`).
- **Thiyaa**: `[Task 8.6]` Build clinician affirmative confirmation modal for setting migrations (`dashboard.js`, `styles.css`).
- **Mithun**: `[Task 8.7]` Implement safety ceiling validator: $P_{\text{plat}} \le 30\text{ cmH}_2\text{O}$ and $\Delta P \le 14\text{ cmH}_2\text{O}$ (`tests/test_pressure_bounds.py`).
- **Ithyaash**: `[Task 8.8]` Implement rate-of-change limiter logic ($|\Delta \text{PEEP}| \le 2, |\Delta FiO_2| \le 0.10$) in `src/adaptive/controller.py`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 8.9]` Generate safety audit compliance report table (`outputs/safety_audit.md`).
- **Thiyaa**: `[Task 8.10]` Style interactive clinician override switch in `dashboard.html`.
- **Mithun**: `[Task 8.11]` Write unit test for emergency manual override fallback state (`tests/test_override.py`).
- **Ithyaash**: `[Task 8.12]` Implement tiered clinician supervision states (Autonomous, Supervised, Manual) in `src/adaptive/controller.py`.

---

### 📅 Week 9: Closed-Loop Physiological Lung Simulator
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 9.1]` Plot 2-compartment compliance and resistance response curves (`scripts/plot_lung_mechanics.py`).
- **Thiyaa**: `[Task 9.2]` Build simulation control sandbox card in `dashboard.html` with Play/Pause/Reset buttons.
- **Mithun**: `[Task 9.3]` Script patient physiological profile generator (normal vs ARDS vs COPD parameters) in `src/adaptive/profiles.py`.
- **Ithyaash**: `[Task 9.4]` Implement 2-compartment mathematical mechanics model in `src/adaptive/simulator.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 9.5]` Generate Berggren shunt equation oxygen content curves (`scripts/plot_shunt_curves.py`).
- **Thiyaa**: `[Task 9.6]` Implement real-time Chart.js dual-axis visualization for simulation telemetry in `dashboard.js`.
- **Mithun**: `[Task 9.7]` Create simulation telemetry export utility to JSON/CSV in `scripts/run_simulation.py`.
- **Ithyaash**: `[Task 9.8]` Implement Berggren shunt equation coupling ($CaO_2$ shunt fraction) in `src/adaptive/simulator.py`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 9.9]` Generate comparative plots: Fixed Settings vs AI Adaptive Strategy during 24h acute lung injury (`scripts/plot_simulation_comparison.py`).
- **Thiyaa**: `[Task 9.10]` Style simulation metrics summary comparison pills (Hypoxemic hours, high driving pressure hours) in `styles.css`.
- **Mithun**: `[Task 9.11]` Write end-to-end simulation reproducibility test (`tests/test_simulation.py`).
- **Ithyaash**: `[Task 9.12]` Finalize closed-loop simulation benchmark in `scripts/run_simulation.py` and tag `v0.3-phase3-complete`.

---

## 🔹 Phase 4: Full System Integration & Academic Paper (Weeks 10 – 12)

### 📅 Week 10: FastAPI Backend & Real-Time Telemetry
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 10.1]` Build statistical summary report of API latency and simulation telemetry bounds (`scripts/report_api_benchmarks.py`).
- **Thiyaa**: `[Task 10.2]` Connect frontend fetch calls to FastAPI REST endpoints in `dashboard.js`.
- **Mithun**: `[Task 10.3]` Implement mock streaming patient feed generator for testing live dashboard (`scripts/mock_stream.py`).
- **Ithyaash**: `[Task 10.4]` Implement FastAPI REST endpoints (`/api/patient/telemetry`, `/api/recommendations`) in `src/web/app.py`.

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 10.5]` Format complete performance comparison matrix for academic paper Section 4 (`docs/paper_tables.md`).
- **Thiyaa**: `[Task 10.6]` Add WebSocket connection handler for real-time telemetry streaming in `dashboard.js`.
- **Mithun**: `[Task 10.7]` Script automated HTTP endpoint regression tests (`tests/test_api.py`).
- **Ithyaash**: `[Task 10.8]` Implement WebSocket telemetry broadcast endpoint in `src/web/app.py`.

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 10.9]` Export high-resolution PNG/SVG vector figures for the academic paper.
- **Thiyaa**: `[Task 10.10]` Polish dark-mode glassmorphic styling, font hierarchy, and responsive breakpoints in `styles.css`.
- **Mithun**: `[Task 10.11]` Create requirements.txt freeze and environment reproducibility audit (`scripts/audit_dependencies.py`).
- **Ithyaash**: `[Task 10.12]` Implement server error-handling middleware and CORS configuration in `src/web/app.py`.

---

### 📅 Week 11: End-to-End System Validation & Paper Drafting
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 11.1]` Complete research paper Section 4 (Results & Performance Analysis) in `research_paper.md`.
- **Thiyaa**: `[Task 11.2]` Create user walkthrough manual with screenshots in `docs/user_manual.md`.
- **Mithun**: `[Task 11.3]` Write automated integration test running extractor, trainer, simulator, and API (`tests/test_full_system.py`).
- **Ithyaash**: `[Task 11.4]` Write research paper Section 3 (Methodology, GroupKFold, and Adaptive Control Architecture).

#### Slot 2 (Days 3–4)
- **Manju**: `[Task 11.5]` Compile and verify all academic citations and clinical literature references in `research_paper.md`.
- **Thiyaa**: `[Task 11.6]` Record UI demo walkthrough video/GIF for repository README.
- **Mithun**: `[Task 11.7]` Audit project license, open-source compliance, and PhysioNet credentialing notice in `README.md`.
- **Ithyaash**: `[Task 11.8]` Write research paper Section 5 (Discussion, Clinical Translation & Governance Roadmap).

#### Slot 3 (Days 5–6)
- **Manju**: `[Task 11.9]` Review paper abstract, keywords, and formatting against target biomedical conference standards.
- **Thiyaa**: `[Task 11.10]` Embed interactive demo media and updated contributor links into `README.md`.
- **Mithun**: `[Task 11.11]` Verify fresh clone and setup workflow on a clean machine (`docs/setup_guide.md`).
- **Ithyaash**: `[Task 11.12]` Conduct complete codebase security audit and code formatting pass (PEP8 / Black).

---

### 📅 Week 12: Final Polish, Demo Rehearsal & Project Defense
#### Slot 1 (Days 1–2)
- **Manju**: `[Task 12.1]` Prepare final presentation slides for academic defense / project exhibition (Clinical problem & Model evaluation).
- **Thiyaa**: `[Task 12.2]` Prepare final presentation slides for system interface, clinician experience, and interactive monitoring station.
- **Mithun**: `[Task 12.3]` Prepare final presentation slides for MIMIC-IV cohort engineering and data pipeline integrity.
- **Ithyaash**: `[Task 12.4]` Prepare final presentation slides for AI control theory, ARDSNet safety boundaries, and technical architecture.

#### Slot 2 (Days 3–4)
- **All Team Members**: `[Task 12.5]` Assemble master slide deck (`docs/presentation_slides.pptx` / PDF) and align presentation speaking slots.
- **All Team Members**: `[Task 12.6]` Conduct full dry-run of live web dashboard demonstration and simulation sandbox execution.

#### Slot 3 (Days 5–6)
- **All Team Members**: `[Task 12.7]` Final review of `README.md`, `INSTRUCTIONS.md`, and `research_paper.md`.
- **Ithyaash**: `[Task 12.8]` Tag production release `v1.0-final-release` and push finalized master repository!
