# 📖 Project Collaboration & Phase Guidelines

> [!IMPORTANT]
> **MANDATORY FOR ALL TEAM MEMBERS (Manju, Thiyaa, Mithun, Ithyaash)**:  
> You **MUST read this entire instruction document** before picking up, working on, or submitting any tasks or Pull Requests.

---

## 🎯 1. Project Overview & Timeline (2.5 – 3 Months)

This project spans **2.5 to 3 Months (10 to 12 Weeks)**. All machine learning models, features, simulation logic, and dashboards will be built **step-by-step from scratch**. 

Tasks are assigned as **daily or alternate-day small issues** to ensure consistent git history, steady progress, and zero burnout.

```
Phase 0 (Week 0): Team Onboarding & Git Workflow Tutorial (Starter Tasks & PRs)

Month 1: Foundation, Data Ingestion & Feature Engineering
├── Phase 1 (Weeks 1–3): Environment Onboarding & MIMIC-IV Cohort Extraction
└── Phase 2A (Weeks 4–5): Feature Engineering & Temporal Rolling Statistics

Month 2: Machine Learning Suite & Physiological Controller
├── Phase 2B (Weeks 6–7): GroupKFold Model Training (Logistic, RF, LightGBM, XGBoost)
└── Phase 3 (Weeks 8–9): ARDSNet Adaptive Controller & Lung Mechanics Simulator

Month 3: Clinical Dashboard, Validation & Academic Paper
├── Phase 4A (Weeks 10–11): Web Monitoring Station & Live Telemetry Integration
└── Phase 4B (Weeks 12): Benchmark Analysis, Research Paper Polish & Final Defense
```

---

## 👥 2. Team Member Focus Areas & Module Ownership

Work is distributed across modular domains, ensuring everyone has meaningful ownership and zero merge conflicts:

| Member | Focus Area | Core Responsibilities & Deliverables |
| :--- | :--- | :--- |
| **Manju** | **Model Evaluation & Visualizations** | Building evaluation plotting scripts (`scripts/plot_results.py`), generating publication-quality ROC/PR curves, confusion matrices, SHAP feature rankings, and statistical performance summaries. |
| **Thiyaa** | **Clinical Interface & User Experience** | Developing the dark-mode monitoring interface (`styles.css`, `dashboard.js`), live Chart.js telemetry graphs, clinical risk badges (Green/Amber/Red), and end-to-end interface sanity tests. |
| **Mithun** | **Data Engineering & Preprocessing** | Temporal rolling aggregations (3h/6h rolling vitals), delta trend slopes ($S/F$ ratio, driving pressure rates of change), missing value imputation strategies, and MIMIC-IV cohort validation scripts. |
| **Ithyaash** | **Predictive Modeling & Control Architecture** | Building the predictive model suite (`src/models/trainer.py`), patient-level GroupKFold cross-validation, LightGBM & XGBoost tuning, loss calibration, ARDSNet bounded adaptive controller math, Berggren shunt equation, and backend integration. |

> 📌 **Day-by-Day Issue Schedule**: For the full list of tasks assigned every 2 days across all 12 weeks, refer to **[`WEEKLY_SCHEDULE.md`](file:///c:/backup/college/Projects/AI-Predictive-and-Adaptive-Ventilator-System-MIMIC-IV/WEEKLY_SCHEDULE.md)**.

---

## 🗓️ 3. Detailed 4-Phase Roadmap (10 – 12 Weeks)

### 🔹 Phase 1: Setup, Data Verification & Environment (Weeks 1 – 3)
* **Goal**: Establish development environments, verify dependencies, and validate the raw MIMIC-IV demo dataset.
* **Tasks**:
  - Environment sanity scripts (`scripts/verify_setup.py`).
  - Auditing ICU stays, demographics, and lab measurement tables.
  - Setting up GitHub branching and PR workflows.

### 🔹 Phase 2: Feature Engineering & Machine Learning from Scratch (Weeks 4 – 7)
* **Goal**: Engineer clinical time-series features and train multi-model predictive benchmarks.
* **Tasks**:
  - **Mithun**: Building temporal rolling statistics (means, standard deviations, delta trends for $SpO_2, FiO_2,$ driving pressure).
  - **Ithyaash**: Implementing `RespiratoryPredictorSuite.train_and_benchmark(...)` with strict patient-level GroupKFold (k=5) to prevent cross-admission leakage.
  - **Ithyaash**: Training & tuning Logistic Regression (baseline), Random Forest, LightGBM, and XGBoost.
  - **Manju**: Extracting and plotting ROC curves, Precision-Recall curves, and feature importance rankings.

### 🔹 Phase 3: Bounded Adaptive Control & Physiological Simulation (Weeks 8 – 9)
* **Goal**: Build the clinical decision-support controller adhering strictly to ARDSNet lung-protective guidelines.
* **Tasks**:
  - **Ithyaash**: Mathematical bounding for tidal volume ($4.0–8.0\text{ mL/kg PBW}$), plateau pressure ($P_{\text{plat}} \le 30\text{ cmH}_2\text{O}$), and rate-of-change limiters.
  - **Ithyaash & Mithun**: Coupling controller outputs to the 2-compartment lung simulator and Berggren oxygen shunt model.
  - **Manju**: Generating comparison plots: Standard Fixed Settings vs. AI Adaptive Strategy during lung injury.

### 🔹 Phase 4: Clinical Web Station & Final Paper Defense (Weeks 10 – 12)
* **Goal**: Deliver a working real-time clinical monitoring station and complete the academic paper.
* **Tasks**:
  - **Thiyaa**: Dark-mode glassmorphic dashboard with live telemetry graphs, risk gauges, and clinician override controls.
  - **Ithyaash**: FastAPI REST and WebSocket endpoints for streaming patient states.
  - **Manju**: Finalizing all publication figures and comparative tables.
  - **All**: Completing `research_paper.md`, rehearsing the final live demo, and preparing presentation slides.

---

## ⚠️ 4. Strict Git & Branching Rules (Zero Direct Push to Main)

- ❌ **NEVER commit or push directly to `main`.**
- ❌ **NEVER merge your own Pull Request into `main`.**
- 🔄 **ALL code changes must go through a dedicated branch and Pull Request (PR).**

### 🛠️ Daily Step-by-Step Git Workflow:

1. **Pull the latest code from `main`**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create a new feature branch for your assigned issue**:
   ```bash
   git checkout -b task/<short-task-name>
   # Example: git checkout -b task/verify-setup
   # Example: git checkout -b task/rolling-features
   # Example: git checkout -b task/plot-roc-curves
   ```

3. **Implement and test locally on your laptop**:
   - Write clean code in your assigned files.
   - Run and test your script or dashboard locally to ensure no syntax errors.

4. **Commit your changes**:
   ```bash
   git add <modified-files>
   git commit -m "feat(<scope>): brief description of what you did"
   ```

5. **Push your branch to GitHub**:
   ```bash
   git push -u origin task/<short-task-name>
   ```

6. **Open a Pull Request (PR)**:
   - On GitHub, open a PR from your branch into `main`.
   - Link your assigned issue in the PR description (e.g., `Closes #1`).

7. **Review & Merge**:
   - The Pull Request will be reviewed for merge status and potential conflicts before being safely merged into `main`.

---

## 💬 5. Communication & Blockers Protocol

- If you run into any issues (library errors, git errors, or questions about the code), **post a comment directly on your assigned GitHub issue**.
- Deadlines are flexible as long as blockers are communicated before the deadline timestamp!
