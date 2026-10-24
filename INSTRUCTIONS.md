# 📖 Project Collaboration & Phase Guidelines

> [!IMPORTANT]
> **MANDATORY FOR ALL TEAM MEMBERS (Manju, Thiyaa, Mithun, Ithyaash)**:  
> You **MUST read this entire instruction document** before picking up, working on, or submitting any tasks or Pull Requests.

---

## 🎯 1. Project Overview & Pacing (2-Month Timeline)

This project spans **2 Months (8 Weeks)** to ensure steady, high-quality development, consistent GitHub activity, and zero burnout. Tasks are assigned as **bite-sized daily or alternate-day issues**.

### 🗓️ Phase Schedule Summary

```
Month 1: Data, Preprocessing & Core AI Modeling
├── Phase 1 (Weeks 1–2): Foundation, Environment Setup & Data Pipeline
└── Phase 2 (Weeks 3–4): Predictive Model Training, GroupKFold & Hyperparameter Tuning

Month 2: Adaptive Control, Dashboard & Academic Paper
├── Phase 3 (Weeks 5–6): ARDSNet Adaptive Controller & Real-Time Web Dashboard
└── Phase 4 (Weeks 7–8): Benchmark Evaluation, Research Paper Polish & Final Defense
```

---

## 👥 2. Team Member Focus Areas

To avoid code clashes and ensure smooth parallel development, work is divided across dedicated domains:

| Member | Domain & Focus Area | Typical Deliverables |
| :--- | :--- | :--- |
| **Manju** | **Model Evaluation & Visualizations** | Plotting ROC/PR curves, confusion matrices, SHAP/feature importance graphs, statistical comparison charts. |
| **Thiyaa** | **Clinical Interface & User Experience** | Dark-mode web dashboard UI (`styles.css`, `dashboard.js`), telemetry widgets, alert badges, and end-to-end interface testing. |
| **Mithun** | **Data Engineering & Cohort Extraction** | MIMIC-IV cohort extraction validation, data sanity scripts (`scripts/`), missing value auditing, and environment checks. |
| **Ithyaash** | **Predictive Modeling & Control Systems** | Machine learning pipelines, ARDSNet adaptive controller algorithms, 2-compartment lung simulation math, and API service. |

---

## 🗓️ 3. Detailed 4-Phase Roadmap

### 🔹 Phase 1: Foundation, Setup & Data Pipeline (Weeks 1 – 2)
* **Goal**: Get all teammates onboarded with Git, verify the local environment on your laptops, and inspect the MIMIC-IV dataset.
* **Key Tasks**:
  - Environment verification and dependencies setup.
  - Verifying extracted MIMIC-IV hourly cohort (`outputs/processed_mimic_cohort.csv`).
  - Exploratory data checks (vital signs ranges, missing lab values).

### 🔹 Phase 2: Core Machine Learning Models (Weeks 3 – 4)
* **Goal**: Train, tune, and evaluate clinical deterioration models.
* **Key Tasks**:
  - Baseline Logistic Regression and Random Forest training.
  - Advanced XGBoost and LightGBM tuning with patient-level GroupKFold (k=5).
  - Generating performance comparison tables (AUROC, AUPRC, Sensitivity, Specificity, Brier score).

### 🔹 Phase 3: Adaptive Controller & Web Dashboard (Weeks 5 – 6)
* **Goal**: Couple the 2-compartment lung simulator to the bounded ARDSNet controller and the interactive web interface.
* **Key Tasks**:
  - Validating clinical safety rules ($P_{\text{plat}} \le 30\text{ cmH}_2\text{O}$, $\Delta P \le 14\text{ cmH}_2\text{O}$, $V_t \in 4–8\text{ mL/kg}$).
  - Live Chart.js telemetry rendering and clinical warning state transitions.
  - Unit tests verifying control output envelopes.

### 🔹 Phase 4: Paper Finalization & Project Defense (Weeks 7 – 8)
* **Goal**: Complete publication-ready documentation and prepare final demo presentation.
* **Key Tasks**:
  - Embedding high-resolution figures into `research_paper.md`.
  - Final code cleanup, license audit, and reproducibility verification.
  - Final rehearsal of live simulation and presentation slide deck.

---

## ⚠️ 4. Strict Git & Branching Rules (Zero Direct Push to Main)

To protect the codebase and avoid overwriting each other's work, **strict Git rules are enforced for everyone**:

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
   - Go to GitHub and open a Pull Request from your branch into `main`.
   - Link your assigned issue in the PR description (e.g., `Closes #1`).

7. **Review & Merge**:
   - The Pull Request will be reviewed for merge status and potential conflicts.
   - Once verified, it will be safely merged into `main`.

---

## 💬 5. Communication & Blockers Protocol

- If you run into any issues (library installation errors, git errors, or questions about the code), **do not stay stuck**.
- **Post a comment directly on your assigned GitHub issue** describing the error or question so the team can help resolve it quickly.
- Deadlines are flexible as long as blockers are communicated before the deadline timestamp!
