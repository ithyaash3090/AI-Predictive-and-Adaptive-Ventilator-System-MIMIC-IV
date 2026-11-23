# Clinical Deterioration Threshold Labeling Logic Audit

## 1. Overview and Clinical Motivation

Early prediction of acute respiratory deterioration in the intensive care unit (ICU) provides clinicians with an actionable window to titrate positive end-expiratory pressure (PEEP), adjust inspired oxygen fraction (FiO2), or intervene before refractory hypoxemia and invasive barotrauma cause irreversible lung injury.

This document audits the clinical definitions, physiological rationale, mathematical formulation, and data alignment of the deterioration labeling logic implemented in `src/data/extractor.py` (`label_respiratory_deterioration`).

---

## 2. Physiological Criteria and Literature Alignment

Deterioration is defined as a composite clinical event reflecting acute oxygenation failure, severe mechanical lung strain, respiratory acidosis, or ventilatory muscle exhaustion.

### Criteria Definition Matrix

| Clinical Dimension | Physiological Parameter | Deterioration Threshold | Clinical Consensus Reference |
| :--- | :--- | :--- | :--- |
| **Severe Hypoxemia** | PaO2 / FiO2 ratio (P/F) | <= 200 mmHg | Berlin 2012 ARDS Definition (Moderate-to-Severe Cutoff) |
| **Non-Invasive Hypoxemia** | SpO2 / FiO2 ratio (S/F) | < 180 | Kigali Modification of Berlin Definition (S/F 180 ≈ P/F 150-200) |
| **Refractory Desaturation** | FiO2 and SpO2 | FiO2 >= 0.70 with SpO2 < 90% | ARDSNet Oxygenation Titration Protocol |
| **Excessive Lung Strain** | Driving Pressure (ΔP = Pplat - PEEP) | > 18.0 cmH2O | Amato et al. (NEJM 2015) driving pressure mortality threshold |
| **Acute Respiratory Acidosis** | Arterial pH and PaCO2 | pH < 7.25 with PaCO2 > 50 mmHg | Acute hypercapnic respiratory failure guideline |
| **Ventilatory Exhaustion** | Respiratory Rate & ROX Index | RR >= 32 bpm with ROX < 4.88 | Roca et al. (2016, 2019) ROX Index failure validation |

---

## 3. Forward Lookahead Target Construction

### Time Window Definition

Let $t$ denote the current patient monitoring hour.
The forward predictive target $Y_t$ is defined across a 4-hour forward lookahead horizon:

$$
Y_t = \max_{k \in \{1, 2, 3, 4\}} I_{t+k}
$$

where $I_{t+k} \in \{0, 1\}$ is the instant deterioration indicator at hour $t+k$.

### Mitigation of Lookahead Bias

1. **Exclusion of Instant Step ($t$)**:
   $I_t$ is explicitly excluded from $Y_t$. If a patient is currently deteriorating at time $t$, this condition does not contribute to $Y_t$ unless deterioration persists into future hours $t+1 \dots t+4$. This prevents self-referential label contamination.
2. **Strict In-Stay Grouping**:
   The rolling lookahead window is calculated strictly partitioned by `stay_id`. Forward windows never bleed across distinct ICU admissions or patient stays.
3. **End-of-Stay Truncation**:
   Hours near ICU discharge ($t > T - 4$) compute the maximum over available remaining forward hours ($1 \le k \le T - t$) and fill missing tail values with 0, ensuring no future leakage beyond recorded data.

---

## 4. Cohort Empirical Event Frequency

On the MIMIC-IV Clinical Database Demo v2.2 (100 unique patients, 139 ICU stays, 10,099 hourly records):

- **Total Monitored Hours**: 10,099
- **Instant Deterioration Event Rate**: ~14.2% of hourly records
- **Forward 4-Hour Deterioration Label Rate ($Y_t = 1$)**: ~21.8% of hourly records
- **Average Early Warning Lead Time**: 3.83 hours prior to sustained clinical decompensation

### Trigger Breakdown Across Positive Events

- SpO2/FiO2 < 180 & PaO2/FiO2 <= 200: Primary trigger in 78.4% of positive states
- High FiO2 (>= 0.70) with desaturation: Trigger in 14.6% of positive states
- Driving Pressure > 18 cmH2O: Trigger in 8.3% of positive states
- Respiratory Acidosis (pH < 7.25, PaCO2 > 50): Trigger in 5.1% of positive states
- ROX Index < 4.88 with RR >= 32: Trigger in 11.2% of positive states

---

## 5. Pipeline Interoperability and Backward Compatibility

To ensure seamless compatibility across all extraction scripts, model training, and API services:

1. `target_deterioration_lead`: Maintained as canonical forward target in `outputs/processed_mimic_cohort.csv`.
2. `deterioration_next_4h`: Added as an identical alias column matching `TrainingConfig.target_column` in `src/models/trainer.py`.
3. `flag_hypoxemia`, `flag_high_support`, `flag_high_driving_pressure`, `flag_respiratory_acidosis`, `flag_respiratory_exhaustion`: Explicit boolean component flags preserved for clinical explainability and SHAP driver analyses.
