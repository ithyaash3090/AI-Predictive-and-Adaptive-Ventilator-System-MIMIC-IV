# Patient-Level Cross-Validation Splitter Audit Report

## 1. Overview
- **Cohort Dataset**: 100 unique patients, 139 ICU stays, 10,099 hourly observations
- **Cross-Validation Scheme**: GroupKFold (k=5) grouped strictly on `subject_id`
- **Evaluation Objective**: Prevent data leakage across multiple admissions/stays of the same patient
- **Leakage Outcome**: Confirmed zero overlapping subjects across all folds

## 2. Fold Partitioning Table

| Fold | Train Patients | Val Patients | Val Stays | Val Rows | Val Positive Count | Val Positive Rate | Leakage Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 80 | 20 | 25 | 2021 | 206 | 10.19% | PASS |
| Fold 2 | 80 | 20 | 29 | 2022 | 400 | 19.78% | PASS |
| Fold 3 | 80 | 20 | 28 | 2019 | 221 | 10.95% | PASS |
| Fold 4 | 80 | 20 | 28 | 2018 | 352 | 17.44% | PASS |
| Fold 5 | 80 | 20 | 29 | 2019 | 214 | 10.60% | PASS |

## 3. Fold Balance Summary
- **Average Positive Rate**: 13.79% (+/- 4.01%)
- **Total Patients Evaluated**: 100 unique individuals
- **Patient Isolation**: Verified strict disjoint sets ($S_{train} \cap S_{val} = \emptyset$ for all folds)
