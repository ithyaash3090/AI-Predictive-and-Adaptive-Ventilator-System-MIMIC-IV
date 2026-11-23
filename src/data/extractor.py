"""
MIMIC-IV Clinical Data Pipeline for Predictive and Adaptive Ventilation System.
Extracts, aligns, and processes ICU stays, chart events, lab events, and diagnoses.
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard Item IDs in MIMIC-IV Chartevents
CHART_ITEMS = {
    220045: ("heart_rate", "bpm"),
    220210: ("respiratory_rate", "insp/min"),
    220277: ("spo2", "%"),
    220052: ("map_art", "mmHg"),
    220181: ("map_nbp", "mmHg"),
    223835: ("fio2", "%"),
    220339: ("peep_set", "cmH2O"),
    224700: ("peep_total", "cmH2O"),
    224684: ("tidal_volume_set", "mL"),
    224685: ("tidal_volume_obs", "mL"),
    224695: ("peak_insp_pressure", "cmH2O"),
    224696: ("plateau_pressure", "cmH2O"),
    224687: ("minute_volume", "L/min"),
}

# Standard Lab Item IDs in MIMIC-IV Labevents
LAB_ITEMS = {
    50821: "pao2",        # Blood Gas pO2 (mmHg)
    50818: "paco2",       # Blood Gas pCO2 (mmHg)
    50820: "ph",          # Blood Gas pH
    50813: "lactate",     # Blood Gas Lactate (mmol/L)
}

class MimicIvExtractor:
    def __init__(self, base_dir="mimic-iv-demo-dataset/mimic-iv-clinical-database-demo-2.2"):
        self.base_dir = base_dir
        self.hosp_dir = os.path.join(base_dir, "hosp")
        self.icu_dir = os.path.join(base_dir, "icu")

    def _csv_path(self, folder, name):
        # MIMIC-IV demo stores table at folder/name.csv/name.csv or folder/name.csv
        p1 = os.path.join(folder, f"{name}.csv", f"{name}.csv")
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(folder, f"{name}.csv")
        if os.path.exists(p2):
            return p2
        raise FileNotFoundError(f"Could not find table {name} in {folder}")

    def load_cohort(self):
        """Loads patients, admissions, and ICU stays."""
        logger.info("Loading patient demographics and ICU admissions...")
        df_patients = pd.read_csv(self._csv_path(self.hosp_dir, "patients"))
        df_admissions = pd.read_csv(self._csv_path(self.hosp_dir, "admissions"))
        df_icustays = pd.read_csv(self._csv_path(self.icu_dir, "icustays"))

        # Datetime conversions
        df_admissions['admittime'] = pd.to_datetime(df_admissions['admittime'])
        df_admissions['dischtime'] = pd.to_datetime(df_admissions['dischtime'])
        df_icustays['intime'] = pd.to_datetime(df_icustays['intime'])
        df_icustays['outtime'] = pd.to_datetime(df_icustays['outtime'])

        # Merge ICU stays with demographics
        cohort = df_icustays.merge(df_patients, on='subject_id', how='inner')
        cohort = cohort.merge(
            df_admissions[['hadm_id', 'admission_type', 'hospital_expire_flag']],
            on='hadm_id',
            how='left'
        )

        # Estimate patient Predicted Body Weight (PBW) based on gender and standard height approximation
        # Male PBW = 50 + 0.91 * (height_cm - 152.4) [Default male ht ~ 175cm -> 70.5kg]
        # Female PBW = 45.5 + 0.91 * (height_cm - 152.4) [Default female ht ~ 162cm -> 54.2kg]
        cohort['pbw_kg'] = np.where(cohort['gender'] == 'M', 70.0, 56.0)
        
        logger.info(f"Loaded cohort: {len(cohort)} ICU stays across {cohort['subject_id'].nunique()} patients.")
        return cohort

    def load_comorbidities(self):
        """Loads ICD diagnoses and maps clinical flags (ARDS, Sepsis, COPD, Pneumonia, Resp Failure)."""
        logger.info("Loading diagnoses and comorbidity flags...")
        df_diag = pd.read_csv(self._csv_path(self.hosp_dir, "diagnoses_icd"))
        df_d_diag = pd.read_csv(self._csv_path(self.hosp_dir, "d_icd_diagnoses"))

        diag_merged = df_diag.merge(df_d_diag, on=['icd_code', 'icd_version'], how='left')
        diag_merged['title_lower'] = diag_merged['long_title'].str.lower().fillna('')

        # Create comorbidity flags by subject_id / hadm_id
        diag_agg = diag_merged.groupby('hadm_id').apply(lambda g: pd.Series({
            'has_respiratory_failure': int(g['title_lower'].str.contains('respiratory failure').any()),
            'has_ards': int(g['title_lower'].str.contains('ards|acute respiratory distress').any()),
            'has_pneumonia': int(g['title_lower'].str.contains('pneumonia').any()),
            'has_copd': int(g['title_lower'].str.contains('chronic obstructive pulmonary|copd|emphysema').any()),
            'has_sepsis': int(g['title_lower'].str.contains('sepsis|septic shock').any()),
            'total_diagnoses': len(g)
        })).reset_index()

        return diag_agg

    def extract_chartevents(self, stay_ids):
        """Extracts vital signs and ventilator parameters for the cohort."""
        logger.info("Extracting chartevents for cohort...")
        csv_path = self._csv_path(self.icu_dir, "chartevents")
        
        chunks = []
        for chunk in pd.read_csv(
            csv_path,
            usecols=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum', 'valueuom'],
            chunksize=100000
        ):
            subset = chunk[chunk['itemid'].isin(CHART_ITEMS.keys()) & chunk['stay_id'].isin(stay_ids)]
            if len(subset) > 0:
                chunks.append(subset)

        if not chunks:
            return pd.DataFrame()

        df_chart = pd.concat(chunks, ignore_index=True)
        df_chart['charttime'] = pd.to_datetime(df_chart['charttime'])
        df_chart['item_name'] = df_chart['itemid'].map(lambda x: CHART_ITEMS[x][0])

        # Normalization and sanity clipping
        # FiO2: sometimes entered as fraction (0.21 - 1.0) and sometimes percentage (21 - 100)
        fio2_mask = df_chart['item_name'] == 'fio2'
        df_chart.loc[fio2_mask & (df_chart['valuenum'] > 1.0), 'valuenum'] = df_chart.loc[fio2_mask & (df_chart['valuenum'] > 1.0), 'valuenum'] / 100.0

        return df_chart

    def extract_labevents(self, subject_ids):
        """Extracts arterial blood gas and metabolic lab values."""
        logger.info("Extracting labevents (blood gases)...")
        csv_path = self._csv_path(self.hosp_dir, "labevents")
        
        chunks = []
        for chunk in pd.read_csv(
            csv_path,
            usecols=['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum'],
            chunksize=100000
        ):
            subset = chunk[chunk['itemid'].isin(LAB_ITEMS.keys()) & chunk['subject_id'].isin(subject_ids)]
            if len(subset) > 0:
                chunks.append(subset)

        if not chunks:
            return pd.DataFrame()

        df_lab = pd.concat(chunks, ignore_index=True)
        df_lab['charttime'] = pd.to_datetime(df_lab['charttime'])
        df_lab['item_name'] = df_lab['itemid'].map(LAB_ITEMS)
        return df_lab

    def build_hourly_timeseries(self, cohort, df_chart, df_lab, df_diag):
        """
        Resamples patient vitals, ventilator parameters, and labs onto hourly intervals.
        Computes derived respiratory metrics (Driving Pressure, SF ratio, PF ratio, ROX Index).
        """
        logger.info("Resampling and constructing hourly aligned clinical timeseries...")
        
        records = []
        for _, stay in cohort.iterrows():
            stay_id = stay['stay_id']
            subject_id = stay['subject_id']
            hadm_id = stay['hadm_id']
            intime = stay['intime']
            outtime = stay['outtime']

            # Patient charts
            stay_chart = df_chart[df_chart['stay_id'] == stay_id]
            if len(stay_chart) == 0:
                continue

            # Pivot chartevents to hourly
            stay_chart = stay_chart.dropna(subset=['valuenum'])
            chart_pivot = stay_chart.pivot_table(
                index='charttime',
                columns='item_name',
                values='valuenum',
                aggfunc='mean'
            )

            # Labs for this patient during ICU window
            stay_labs = df_lab[
                (df_lab['subject_id'] == subject_id) & 
                (df_lab['charttime'] >= intime - timedelta(hours=6)) & 
                (df_lab['charttime'] <= outtime + timedelta(hours=2))
            ]
            if len(stay_labs) > 0:
                lab_pivot = stay_labs.pivot_table(
                    index='charttime',
                    columns='item_name',
                    values='valuenum',
                    aggfunc='mean'
                )
            else:
                lab_pivot = pd.DataFrame()

            # Combine chart and lab onto continuous hourly grid
            combined = pd.concat([chart_pivot, lab_pivot], axis=1)
            hourly = combined.resample('1h').mean()

            # Restrict to stay bounds with some margin
            start_hr = intime.floor('h')
            end_hr = min(outtime.ceil('h'), start_hr + timedelta(hours=168)) # max 7 days for demo
            hourly = hourly[(hourly.index >= start_hr) & (hourly.index <= end_hr)]

            if len(hourly) < 3: # require at least 3 hours of monitoring
                continue

            # Forward fill up to 4 hours for physiological signals, then backfill
            hourly = hourly.ffill(limit=6).bfill(limit=2)

            # Consolidate duplicate/alternate columns
            # MAP: coalesce arterial and non-invasive
            map_val = hourly.get('map_art', pd.Series(index=hourly.index, dtype=float)).combine_first(
                hourly.get('map_nbp', pd.Series(index=hourly.index, dtype=float))
            )
            # PEEP: coalesce set and total
            peep_val = hourly.get('peep_set', pd.Series(index=hourly.index, dtype=float)).combine_first(
                hourly.get('peep_total', pd.Series(index=hourly.index, dtype=float))
            )
            # Tidal Volume: coalesce observed and set
            vt_val = hourly.get('tidal_volume_obs', pd.Series(index=hourly.index, dtype=float)).combine_first(
                hourly.get('tidal_volume_set', pd.Series(index=hourly.index, dtype=float))
            )

            hourly['mean_arterial_pressure'] = map_val
            hourly['peep'] = peep_val
            hourly['tidal_volume'] = vt_val

            # Derived clinical metrics
            # 1. Driving Pressure = Plateau Pressure - PEEP
            if 'plateau_pressure' in hourly.columns and 'peep' in hourly.columns:
                hourly['driving_pressure'] = (hourly['plateau_pressure'] - hourly['peep']).clip(lower=0, upper=40)
            else:
                hourly['driving_pressure'] = np.nan

            # 2. SF Ratio = SpO2 / FiO2 (Non-invasive surrogate for PF ratio)
            if 'spo2' in hourly.columns and 'fio2' in hourly.columns:
                hourly['sf_ratio'] = hourly['spo2'] / (hourly['fio2'].clip(lower=0.21, upper=1.0))
            else:
                hourly['sf_ratio'] = np.nan

            # 3. PF Ratio = PaO2 / FiO2 (Standard Berlin definition of ARDS)
            if 'pao2' in hourly.columns and 'fio2' in hourly.columns:
                hourly['pf_ratio'] = hourly['pao2'] / (hourly['fio2'].clip(lower=0.21, upper=1.0))
            else:
                hourly['pf_ratio'] = np.nan

            # 4. ROX Index = (SpO2 / FiO2) / Respiratory Rate
            if 'sf_ratio' in hourly.columns and 'respiratory_rate' in hourly.columns:
                hourly['rox_index'] = (hourly['sf_ratio'] / hourly['respiratory_rate'].clip(lower=5)).clip(lower=1, upper=40)
            else:
                hourly['rox_index'] = np.nan

            # 5. Ventilatory Ratio / Normalized Tidal Volume by PBW
            pbw = stay['pbw_kg']
            hourly['vt_per_kg_pbw'] = hourly['tidal_volume'] / pbw

            # Attach metadata
            hourly['stay_id'] = stay_id
            hourly['subject_id'] = subject_id
            hourly['hadm_id'] = hadm_id
            hourly['age'] = stay['anchor_age']
            hourly['gender'] = stay['gender']
            hourly['pbw_kg'] = pbw
            hourly['hour_from_intime'] = ((hourly.index - start_hr).total_seconds() / 3600).astype(int)

            records.append(hourly.reset_index().rename(columns={'charttime': 'timestamp'}))

        if not records:
            raise ValueError("No valid patient timeseries extracted.")

        full_ts = pd.concat(records, ignore_index=True)

        # Merge comorbidities
        full_ts = full_ts.merge(df_diag, on='hadm_id', how='left')
        comorb_cols = ['has_respiratory_failure', 'has_ards', 'has_pneumonia', 'has_copd', 'has_sepsis']
        full_ts[comorb_cols] = full_ts[comorb_cols].fillna(0).astype(int)

        logger.info(f"Successfully constructed hourly timeseries: {len(full_ts)} hourly records across {full_ts['stay_id'].nunique()} ICU stays.")
        return full_ts

    def label_respiratory_deterioration(self, df_ts: pd.DataFrame, lookahead_hours: int = 4) -> pd.DataFrame:
        """
        Defines clinically grounded Respiratory Deterioration event aligned with Berlin 2012 ARDS Definition:
        
        A patient hour t experiences an 'instant deterioration' if any of the following 5 clinical criteria trigger:
        1. Severe Hypoxemia:
           - PaO2/FiO2 <= 200 mmHg (Moderate-to-Severe ARDS cutoff per Berlin Definition)
           - OR SpO2/FiO2 < 180 (Non-invasive validated surrogate for P/F <= 150-200)
        2. Escalation Requirement:
           - FiO2 >= 0.70 with concurrent SpO2 < 90% (refractory hypoxemia despite high fraction)
        3. Ventilatory / Lung Injury Danger:
           - Driving pressure (Pplat - PEEP) > 18.0 cmH2O (excessive transpulmonary mechanical stress)
           - OR Acute Respiratory Acidosis (arterial pH < 7.25 and PaCO2 > 50.0 mmHg)
        4. Tachypneic Exhaustion / Impending Ventilatory Failure:
           - Respiratory Rate >= 32 breaths/min with ROX Index < 4.88 (validated intubation/failure threshold)

        Lookahead Target Construction:
        The forward predictive target for hour t is 1 if ANY instant deterioration occurs strictly in
        the forward window [t + 1, t + lookahead_hours], and 0 otherwise.
        Hour t itself is strictly excluded from its own forward label to prevent lookahead data leakage.
        """
        logger.info(f"Labeling respiratory deterioration events (lookahead={lookahead_hours}h)...")
        
        # 1. Clinical condition flags at current step
        hypoxemia = (df_ts['sf_ratio'] < 180.0) | (df_ts['pf_ratio'] <= 200.0)
        high_support = (df_ts['fio2'] >= 0.70) & (df_ts['spo2'] < 90.0)
        high_driving_p = df_ts['driving_pressure'] > 18.0
        resp_acidosis = (df_ts['ph'] < 7.25) & (df_ts['paco2'] > 50.0)
        resp_exhaustion = (df_ts['respiratory_rate'] >= 32.0) & (df_ts['rox_index'] < 4.88)

        # Store individual component flags for auditability
        df_ts['flag_hypoxemia'] = hypoxemia.fillna(False).astype(int)
        df_ts['flag_high_support'] = high_support.fillna(False).astype(int)
        df_ts['flag_high_driving_pressure'] = high_driving_p.fillna(False).astype(int)
        df_ts['flag_respiratory_acidosis'] = resp_acidosis.fillna(False).astype(int)
        df_ts['flag_respiratory_exhaustion'] = resp_exhaustion.fillna(False).astype(int)

        instant_deterioration = (
            hypoxemia | high_support | high_driving_p | resp_acidosis | resp_exhaustion
        ).fillna(False).astype(int)
        df_ts['instant_deterioration'] = instant_deterioration

        # 2. Lookahead target: max of instant deterioration in forward window [t+1, t+lookahead_hours]
        df_ts = df_ts.sort_values(['stay_id', 'hour_from_intime']).reset_index(drop=True)
        
        def compute_forward_target(s: pd.Series) -> pd.Series:
            # Reverse series so rolling window looks into future
            rev_events = s.iloc[::-1]
            # Shift by 1 step to exclude current time t (strictly t+1 to t+lookahead_hours)
            return rev_events.rolling(window=lookahead_hours, min_periods=1).max().shift(1).iloc[::-1]

        rolled = df_ts.groupby('stay_id')['instant_deterioration'].transform(compute_forward_target)
        df_ts['target_deterioration_lead'] = rolled.fillna(0).astype(int)
        df_ts['deterioration_next_4h'] = df_ts['target_deterioration_lead']
        df_labeled = df_ts
        
        pos_rate = df_labeled['target_deterioration_lead'].mean()
        logger.info(f"Deterioration labeling complete. Total hours: {len(df_labeled)}, Positive event rate: {pos_rate:.2%}")
        return df_labeled

    @staticmethod
    def audit_label_distribution(df_labeled: pd.DataFrame) -> dict:
        """
        Computes detailed statistical audit of deterioration labels and criteria breakdown.
        """
        total_rows = len(df_labeled)
        total_stays = df_labeled['stay_id'].nunique() if 'stay_id' in df_labeled.columns else 0
        total_patients = df_labeled['subject_id'].nunique() if 'subject_id' in df_labeled.columns else 0

        target_col = 'target_deterioration_lead' if 'target_deterioration_lead' in df_labeled.columns else 'deterioration_next_4h'
        pos_count = int(df_labeled[target_col].sum()) if target_col in df_labeled.columns else 0
        neg_count = total_rows - pos_count

        flags = {
            'hypoxemia': int(df_labeled['flag_hypoxemia'].sum()) if 'flag_hypoxemia' in df_labeled.columns else 0,
            'high_support': int(df_labeled['flag_high_support'].sum()) if 'flag_high_support' in df_labeled.columns else 0,
            'high_driving_pressure': int(df_labeled['flag_high_driving_pressure'].sum()) if 'flag_high_driving_pressure' in df_labeled.columns else 0,
            'respiratory_acidosis': int(df_labeled['flag_respiratory_acidosis'].sum()) if 'flag_respiratory_acidosis' in df_labeled.columns else 0,
            'respiratory_exhaustion': int(df_labeled['flag_respiratory_exhaustion'].sum()) if 'flag_respiratory_exhaustion' in df_labeled.columns else 0,
        }

        return {
            'total_rows': total_rows,
            'total_stays': total_stays,
            'total_patients': total_patients,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'positive_prevalence': float(pos_count / total_rows) if total_rows > 0 else 0.0,
            'criteria_breakdown': flags,
        }
