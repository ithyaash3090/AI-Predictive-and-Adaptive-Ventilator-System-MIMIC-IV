/**
 * AI-Powered Adaptive & Predictive Ventilation Support System
 * Interactive Clinical Telemetry & Dashboard Logic
 */

let currentStayId = null;
let currentPatientData = null;
let timeseriesChart = null;
let rocChart = null;
let simSpo2Chart = null;
let simDpChart = null;
let autonomousMode = false;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initLiveClock();
  loadPatients();
  loadBenchmarks();
  loadSimulation();
});

function initLiveClock() {
  const clockEl = document.getElementById("live-clock");
  setInterval(() => {
    const now = new Date();
    clockEl.innerText = now.toISOString().replace("T", " ").substring(0, 19) + " UTC";
  }, 1000);
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
  document.querySelectorAll(".nav-tab-btn").forEach(btn => btn.classList.remove("active"));

  const targetPane = document.getElementById(`pane-${tabId}`);
  const targetBtn = document.getElementById(`tab-btn-${tabId}`);
  if (targetPane) targetPane.classList.add("active");
  if (targetBtn) targetBtn.classList.add("active");
}

function showToast(message) {
  const toast = document.getElementById("toast");
  const msgEl = document.getElementById("toast-message");
  msgEl.innerText = message;
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}

// -------------------------------------------------------------
// MODULE 1: Patient Data Loading & Telemetry Rendering
// -------------------------------------------------------------
async function loadPatients() {
  try {
    const res = await fetch("/api/patients");
    const data = await res.json();
    const select = document.getElementById("patient-select");
    select.innerHTML = "";

    data.patients.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.stay_id;
      const statusIcon = p.experienced_deterioration ? "⚠️ [HIGH RISK]" : "✅ [STABLE]";
      opt.innerText = `${statusIcon} Stay #${p.stay_id} | Subj #${p.subject_id} (${p.age}y, ${p.gender}) - ${p.primary_diagnoses} [Peak Risk: ${(p.max_risk * 100).toFixed(0)}%]`;
      select.appendChild(opt);
    });

    if (data.patients.length > 0) {
      select.value = data.patients[0].stay_id;
      onPatientChange(data.patients[0].stay_id);
    }
  } catch (err) {
    console.error("Error loading patients:", err);
  }
}

async function onPatientChange(stayId) {
  if (!stayId) return;
  currentStayId = stayId;

  try {
    const res = await fetch(`/api/patient/${stayId}`);
    const data = await res.json();
    currentPatientData = data;

    renderPatientDemographics(data.demographics);
    renderLongitudinalTelemetry(data.timeseries);
  } catch (err) {
    console.error("Error fetching patient details:", err);
  }
}

function renderPatientDemographics(demo) {
  document.getElementById("badge-patient-id").innerText = `Subject ID: ${demo.subject_id}`;
  document.getElementById("badge-age-gender").innerText = `${demo.age} Yrs | ${demo.gender === 'M' ? 'Male' : 'Female'}`;
  document.getElementById("badge-pbw").innerText = `PBW: ${demo.pbw_kg.toFixed(1)} kg`;

  const diags = [];
  if (demo.has_ards) diags.push("ARDS");
  if (demo.has_respiratory_failure) diags.push("Resp Failure");
  if (demo.has_pneumonia) diags.push("Pneumonia");
  if (demo.has_copd) diags.push("COPD");
  if (demo.has_sepsis) diags.push("Sepsis");

  document.getElementById("badge-diagnoses").innerText = diags.length > 0 ? diags.join(", ") : "ICU Admission";
}

function renderLongitudinalTelemetry(ts) {
  if (!ts || ts.length === 0) return;

  // Most recent or peak deteriorated step to showcase clinical decision support
  const deterioratedSteps = ts.filter(r => r.instant_deterioration === 1 || r.target_deterioration_lead === 1);
  const currentStep = deterioratedSteps.length > 0 ? deterioratedSteps[0] : ts[ts.length - 1];

  // 1. Update Telemetry Tiles
  const spo2 = currentStep.spo2 != null ? currentStep.spo2.toFixed(1) : "95.0";
  const fio2 = currentStep.fio2 != null ? currentStep.fio2.toFixed(2) : "0.35";
  const peep = currentStep.peep != null ? currentStep.peep.toFixed(1) : "5.0";
  const dp = currentStep.driving_pressure != null ? currentStep.driving_pressure.toFixed(1) : "11.0";
  const vt = currentStep.tidal_volume != null ? Math.round(currentStep.tidal_volume) : 450;
  const rr = currentStep.respiratory_rate != null ? Math.round(currentStep.respiratory_rate) : 18;
  const rox = currentStep.rox_index != null ? currentStep.rox_index.toFixed(1) : "14.2";
  const mapVal = currentStep.mean_arterial_pressure != null ? Math.round(currentStep.mean_arterial_pressure) : 75;

  document.getElementById("val-spo2").innerHTML = `${spo2} <span class="tile-unit">%</span>`;
  document.getElementById("val-fio2").innerHTML = `${fio2} <span class="tile-unit"></span>`;
  document.getElementById("val-peep").innerHTML = `${peep} <span class="tile-unit">cmH2O</span>`;
  
  const dpEl = document.getElementById("val-dp");
  dpEl.innerHTML = `${dp} <span class="tile-unit">cmH2O</span>`;
  dpEl.style.color = parseFloat(dp) > 14.0 ? "var(--accent-crimson)" : "var(--accent-emerald)";

  document.getElementById("val-vt").innerHTML = `${vt} <span class="tile-unit">mL</span>`;
  document.getElementById("val-vt-kg").innerText = `${(vt / currentPatientData.demographics.pbw_kg).toFixed(1)} mL/kg PBW`;
  document.getElementById("val-rr").innerHTML = `${rr} <span class="tile-unit">bpm</span>`;
  document.getElementById("val-rox").innerHTML = `${rox} <span class="tile-unit"></span>`;
  document.getElementById("val-map").innerHTML = `${mapVal} <span class="tile-unit">mmHg</span>`;

  // 2. Predictive Risk Gauge
  const risk = currentStep.predicted_risk_xgb != null ? currentStep.predicted_risk_xgb : 0.20;
  updateRiskGauge(risk);

  // 3. Clinical Risk Drivers
  renderRiskDrivers(currentStep);

  // 4. Longitudinal Timeseries Chart
  renderChart(ts);

  // 5. Query Adaptive Recommendation
  fetchRecommendation({
    fio2: parseFloat(fio2),
    peep: parseFloat(peep),
    tidal_volume_ml: parseFloat(vt),
    respiratory_rate: parseInt(rr),
    spo2: parseFloat(spo2),
    driving_pressure: parseFloat(dp),
    plateau_pressure: currentStep.plateau_pressure || (parseFloat(peep) + parseFloat(dp)),
    ph: currentStep.ph || 7.40,
    paco2: currentStep.paco2 || 40.0,
    pbw_kg: currentPatientData.demographics.pbw_kg,
    predicted_risk: risk,
    is_ards_or_copd: currentPatientData.demographics.has_ards || currentPatientData.demographics.has_copd
  });
}

function updateRiskGauge(risk) {
  const percent = Math.round(risk * 100);
  const displayEl = document.getElementById("risk-score-display");
  const badgeEl = document.getElementById("risk-level-badge");
  const circleEl = document.getElementById("risk-circle-prog");

  displayEl.innerText = `${percent}%`;

  // Circumference of r=50 circle is ~314.15
  const offset = 314.15 - (314.15 * risk);
  circleEl.style.strokeDashoffset = offset;

  if (risk < 0.25) {
    displayEl.style.color = "#10b981";
    circleEl.style.stroke = "#10b981";
    badgeEl.className = "badge badge-emerald";
    badgeEl.innerText = "NORMAL";
  } else if (risk < 0.50) {
    displayEl.style.color = "#3b82f6";
    circleEl.style.stroke = "#3b82f6";
    badgeEl.className = "badge badge-blue";
    badgeEl.innerText = "GUARDED";
  } else if (risk < 0.75) {
    displayEl.style.color = "#f59e0b";
    circleEl.style.stroke = "#f59e0b";
    badgeEl.className = "badge badge-amber";
    badgeEl.innerText = "HIGH RISK";
  } else {
    displayEl.style.color = "#ef4444";
    circleEl.style.stroke = "#ef4444";
    badgeEl.className = "badge badge-crimson";
    badgeEl.innerText = "CRITICAL DETERIORATION";
  }
}

function renderRiskDrivers(step) {
  const container = document.getElementById("patient-risk-drivers");
  container.innerHTML = "";

  const drivers = [];
  if (step.sf_ratio && step.sf_ratio < 220) {
    drivers.push({ name: "SpO2/FiO2 Ratio Decline", val: `${step.sf_ratio.toFixed(0)} (Target > 300)`, severity: "high" });
  }
  if (step.driving_pressure && step.driving_pressure > 14) {
    drivers.push({ name: "Driving Pressure Strain", val: `${step.driving_pressure.toFixed(1)} cmH2O (>14.0)`, severity: "high" });
  }
  if (step.fio2 && step.fio2 >= 0.50) {
    drivers.push({ name: "Elevated Oxygen Fraction", val: `FiO2 ${(step.fio2 * 100).toFixed(0)}%`, severity: "medium" });
  }
  if (step.rox_index && step.rox_index < 4.88) {
    drivers.push({ name: "Low ROX Index", val: `${step.rox_index.toFixed(2)} (<4.88 exhaustion)`, severity: "high" });
  }
  if (step.lactate && step.lactate > 2.0) {
    drivers.push({ name: "Elevated Arterial Lactate", val: `${step.lactate.toFixed(1)} mmol/L`, severity: "medium" });
  }

  if (drivers.length === 0) {
    drivers.push({ name: "Physiological Metrics in Range", val: "Optimal oxygenation & mechanics", severity: "normal" });
  }

  drivers.forEach(d => {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.justifyContent = "space-between";
    const color = d.severity === "high" ? "#f87171" : d.severity === "medium" ? "#fbbf24" : "#34d399";
    row.innerHTML = `<span style="color: var(--text-secondary);">${d.name}:</span><strong style="color: ${color}; font-family: var(--font-mono);">${d.val}</strong>`;
    container.appendChild(row);
  });
}

function renderChart(timeseries) {
  const ctx = document.getElementById("patient-timeseries-chart").getContext("2d");
  const labels = timeseries.map(r => `Hr ${r.hour_from_intime}`);
  const spo2Data = timeseries.map(r => r.spo2);
  const fio2Percent = timeseries.map(r => r.fio2 != null ? (r.fio2 * 100) : null);
  const peepData = timeseries.map(r => r.peep);
  const dpData = timeseries.map(r => r.driving_pressure);

  if (timeseriesChart) timeseriesChart.destroy();

  timeseriesChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "SpO2 (%)",
          data: spo2Data,
          borderColor: "#00f2fe",
          backgroundColor: "rgba(0, 242, 254, 0.08)",
          borderWidth: 2,
          yAxisID: "y",
          tension: 0.3
        },
        {
          label: "FiO2 (%)",
          data: fio2Percent,
          borderColor: "#3b82f6",
          borderWidth: 2,
          borderDash: [4, 4],
          yAxisID: "y",
          tension: 0.2
        },
        {
          label: "PEEP (cmH2O)",
          data: peepData,
          borderColor: "#8b5cf6",
          borderWidth: 2,
          yAxisID: "y1",
          tension: 0.2
        },
        {
          label: "Driving Pressure (cmH2O)",
          data: dpData,
          borderColor: "#ef4444",
          borderWidth: 2,
          yAxisID: "y1",
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 11 } } },
        tooltip: { mode: "index", intersect: false }
      },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 12 }, grid: { color: "rgba(255,255,255,0.03)" } },
        y: {
          min: 60,
          max: 100,
          ticks: { color: "#00f2fe" },
          grid: { color: "rgba(255,255,255,0.04)" },
          title: { display: true, text: "Oxygenation (%)", color: "#00f2fe" }
        },
        y1: {
          position: "right",
          min: 0,
          max: 30,
          ticks: { color: "#ef4444" },
          grid: { drawOnChartArea: false },
          title: { display: true, text: "Pressures (cmH2O)", color: "#ef4444" }
        }
      }
    }
  });
}

// -------------------------------------------------------------
// MODULE 2: Adaptive Recommendations & Safety Constraints
// -------------------------------------------------------------
async function fetchRecommendation(payload) {
  try {
    const res = await fetch("/api/adaptive/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const rec = data.recommendation;

    // Update settings comparison table
    document.getElementById("rec-curr-fio2").innerText = rec.current_settings.fio2.toFixed(2);
    document.getElementById("rec-prop-fio2").innerText = rec.proposed_settings.fio2.toFixed(2);
    formatDelta("rec-delta-fio2", rec.setting_deltas.delta_fio2, "");

    document.getElementById("rec-curr-peep").innerText = rec.current_settings.peep.toFixed(1);
    document.getElementById("rec-prop-peep").innerText = rec.proposed_settings.peep.toFixed(1);
    formatDelta("rec-delta-peep", rec.setting_deltas.delta_peep, "");

    document.getElementById("rec-curr-vt").innerText = Math.round(rec.current_settings.tidal_volume_ml);
    document.getElementById("rec-prop-vt").innerText = Math.round(rec.proposed_settings.tidal_volume_ml);
    formatDelta("rec-delta-vt", rec.setting_deltas.delta_tidal_volume_ml, "");

    document.getElementById("rec-curr-rr").innerText = rec.current_settings.respiratory_rate;
    document.getElementById("rec-prop-rr").innerText = rec.proposed_settings.respiratory_rate;
    formatDelta("rec-delta-rr", rec.setting_deltas.delta_respiratory_rate, "");

    // Safety checks
    updateCheckIcon("chk-plateau", rec.safety_checks.plateau_limit_respected);
    updateCheckIcon("chk-driving", rec.proposed_settings.driving_pressure ? rec.proposed_settings.driving_pressure <= 14 : true);
    updateCheckIcon("chk-vt", rec.safety_checks.vt_per_kg_safe);
    updateCheckIcon("chk-step", rec.safety_checks.max_step_fio2_passed && rec.safety_checks.max_step_peep_passed);

    // Rationale
    document.getElementById("rec-rationale-box").innerHTML = rec.clinical_rationale.map(r => `• ${r}`).join("<br>");

  } catch (err) {
    console.error("Error fetching recommendation:", err);
  }
}

function formatDelta(elId, val, unit) {
  const el = document.getElementById(elId);
  if (val > 0) {
    el.className = "delta-badge delta-pos";
    el.innerText = `+${val}${unit}`;
  } else if (val < 0) {
    el.className = "delta-badge delta-neg";
    el.innerText = `${val}${unit}`;
  } else {
    el.className = "delta-badge delta-zero";
    el.innerText = `0${unit}`;
  }
}

function updateCheckIcon(elId, passed) {
  const el = document.getElementById(elId);
  if (passed) {
    el.className = "check-icon check-pass";
    el.innerHTML = "&check;";
  } else {
    el.className = "check-icon check-fail";
    el.innerHTML = "&times;";
  }
}

async function applyRecommendation(actionType) {
  if (!currentStayId) return;

  const propFio2 = parseFloat(document.getElementById("rec-prop-fio2").innerText);
  const propPeep = parseFloat(document.getElementById("rec-prop-peep").innerText);
  const propVt = parseFloat(document.getElementById("rec-prop-vt").innerText);
  const propRr = parseInt(document.getElementById("rec-prop-rr").innerText);

  try {
    const res = await fetch("/api/action/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stay_id: parseInt(currentStayId),
        action_type: actionType,
        applied_settings: { fio2: propFio2, peep: propPeep, tidal_volume_ml: propVt, respiratory_rate: propRr },
        clinician_notes: `Clinician triggered ${actionType} at bed station.`
      })
    });
    const result = await res.json();
    showToast(result.message);
  } catch (err) {
    console.error("Error logging action:", err);
  }
}

function toggleAutonomousMode() {
  autonomousMode = !autonomousMode;
  const statusEl = document.getElementById("auto-mode-status");
  if (autonomousMode) {
    statusEl.innerText = "ACTIVE (SIM)";
    statusEl.style.color = "var(--accent-cyan)";
    showToast("Autonomous research mode enabled in controlled sandbox.");
  } else {
    statusEl.innerText = "OFF";
    statusEl.style.color = "var(--accent-emerald)";
    showToast("Autonomous mode disabled. Clinician sign-off required.");
  }
}

function openModifyModal() {
  showToast("Modify Settings Panel: Clinician parameter slider active.");
}

// -------------------------------------------------------------
// MODULE 3: Benchmarks & Explainability
// -------------------------------------------------------------
async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();
    if (!data || !data["XGBoost"]) return;

    // 1. Populate table
    const tbody = document.getElementById("benchmark-table-body");
    tbody.innerHTML = "";

    const models = ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"];
    models.forEach(mName => {
      const m = data[mName];
      if (!m) return;
      const row = document.createElement("tr");
      const isXgb = mName === "XGBoost";
      row.innerHTML = `
        <td style="font-weight: 600; color: #fff;">${mName}</td>
        <td class="${isXgb ? 'highlight-best' : ''}">${m.auroc.toFixed(4)} &plusmn; ${m.auroc_std}</td>
        <td class="${isXgb ? 'highlight-best' : ''}">${m.auprc.toFixed(4)} &plusmn; ${m.auprc_std}</td>
        <td>${(m.sensitivity * 100).toFixed(1)}%</td>
        <td class="${isXgb ? 'highlight-best' : ''}">${(m.specificity * 100).toFixed(1)}%</td>
        <td>${(m.precision * 100).toFixed(1)}%</td>
        <td>${m.f1.toFixed(4)}</td>
        <td>${m.brier.toFixed(4)}</td>
      `;
      tbody.appendChild(row);
    });

    // 2. Render ROC Curves
    renderRocChart(data);

    // 3. Render Top Features
    renderFeatureImportance(data.top_features);

  } catch (err) {
    console.error("Error loading benchmark:", err);
  }
}

function renderRocChart(data) {
  const ctx = document.getElementById("roc-chart").getContext("2d");
  const xgbRoc = data["XGBoost"] ? data["XGBoost"].roc_curve : null;
  const lgbRoc = data["LightGBM"] ? data["LightGBM"].roc_curve : null;
  const rfRoc = data["Random Forest"] ? data["Random Forest"].roc_curve : null;
  const lrRoc = data["Logistic Regression"] ? data["Logistic Regression"].roc_curve : null;

  if (rocChart) rocChart.destroy();

  const datasets = [];
  if (xgbRoc) {
    datasets.push({
      label: `XGBoost (AUC = ${data["XGBoost"].auroc})`,
      data: xgbRoc.fpr.map((x, i) => ({ x: x, y: xgbRoc.tpr[i] })),
      borderColor: "#00f2fe",
      borderWidth: 2.5,
      tension: 0.1
    });
  }
  if (lgbRoc) {
    datasets.push({
      label: `LightGBM (AUC = ${data["LightGBM"].auroc})`,
      data: lgbRoc.fpr.map((x, i) => ({ x: x, y: lgbRoc.tpr[i] })),
      borderColor: "#3b82f6",
      borderWidth: 2,
      tension: 0.1
    });
  }
  if (rfRoc) {
    datasets.push({
      label: `Random Forest (AUC = ${data["Random Forest"].auroc})`,
      data: rfRoc.fpr.map((x, i) => ({ x: x, y: rfRoc.tpr[i] })),
      borderColor: "#10b981",
      borderWidth: 1.5,
      borderDash: [3, 3]
    });
  }
  if (lrRoc) {
    datasets.push({
      label: `Logistic Regression (AUC = ${data["Logistic Regression"].auroc})`,
      data: lrRoc.fpr.map((x, i) => ({ x: x, y: lrRoc.tpr[i] })),
      borderColor: "#94a3b8",
      borderWidth: 1.5,
      borderDash: [5, 5]
    });
  }

  rocChart = new Chart(ctx, {
    type: "line",
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 10 } } }
      },
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#64748b" }, ticks: { color: "#64748b" } },
        y: { type: "linear", min: 0, max: 1, title: { display: true, text: "True Positive Rate (Sensitivity)", color: "#64748b" }, ticks: { color: "#64748b" } }
      }
    }
  });
}

function renderFeatureImportance(features) {
  const container = document.getElementById("feature-importance-bars");
  container.innerHTML = "";
  if (!features || features.length === 0) return;

  const maxImp = features[0].importance;

  features.slice(0, 10).forEach(f => {
    const row = document.createElement("div");
    row.className = "feature-bar-row";
    const percent = ((f.importance / maxImp) * 100).toFixed(0);
    row.innerHTML = `
      <div class="feature-name" title="${f.feature}">${f.feature}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${percent}%;"></div>
      </div>
      <div class="bar-value">${f.importance.toFixed(3)}</div>
    `;
    container.appendChild(row);
  });
}

// -------------------------------------------------------------
// MODULE 4: Physiological Respiratory Simulator
// -------------------------------------------------------------
async function loadSimulation() {
  await runLiveSimulation();
}

async function runLiveSimulation() {
  const duration = parseInt(document.getElementById("slider-duration").value);

  try {
    const res = await fetch("/api/simulation/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ total_steps: duration })
    });
    const data = await res.json();

    renderSimulationCharts(data);
    renderSimulationSummary(data.summary);
    showToast("Simulation execution completed.");
  } catch (err) {
    console.error("Error executing simulation:", err);
  }
}

function renderSimulationCharts(data) {
  const steps = data.fixed_history.map(r => `Hr ${r.hour}`);
  const fixedSpo2 = data.fixed_history.map(r => r.vitals.spo2);
  const adaptSpo2 = data.adaptive_history.map(r => r.vitals.spo2);

  const fixedDp = data.fixed_history.map(r => r.vitals.driving_pressure);
  const adaptDp = data.adaptive_history.map(r => r.vitals.driving_pressure);

  // SpO2 Chart
  const ctxSpo2 = document.getElementById("sim-spo2-chart").getContext("2d");
  if (simSpo2Chart) simSpo2Chart.destroy();

  simSpo2Chart = new Chart(ctxSpo2, {
    type: "line",
    data: {
      labels: steps,
      datasets: [
        {
          label: "Fixed Settings Arm (Unadjusted)",
          data: fixedSpo2,
          borderColor: "#ef4444",
          borderWidth: 2.5,
          tension: 0.3
        },
        {
          label: "AI-Adaptive Arm (ARDSNet Bounded)",
          data: adaptSpo2,
          borderColor: "#10b981",
          borderWidth: 2.5,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 80, max: 100, title: { display: true, text: "SpO2 (%)", color: "#94a3b8" } }
      }
    }
  });

  // Driving Pressure Chart
  const ctxDp = document.getElementById("sim-dp-chart").getContext("2d");
  if (simDpChart) simDpChart.destroy();

  simDpChart = new Chart(ctxDp, {
    type: "line",
    data: {
      labels: steps,
      datasets: [
        {
          label: "Fixed Settings Driving Pressure",
          data: fixedDp,
          borderColor: "#ef4444",
          borderWidth: 2.5,
          tension: 0.3
        },
        {
          label: "AI-Adaptive Driving Pressure",
          data: adaptDp,
          borderColor: "#10b981",
          borderWidth: 2.5,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 8, max: 24, title: { display: true, text: "Driving Pressure ΔP (cmH2O)", color: "#94a3b8" } }
      }
    }
  });
}

function renderSimulationSummary(summary) {
  const f = summary.fixed_strategy;
  const a = summary.adaptive_strategy;

  document.getElementById("sim-fixed-min-spo2").innerText = `${f.min_spo2}%`;
  document.getElementById("sim-fixed-max-dp").innerText = `${f.max_driving_pressure} cmH2O`;
  document.getElementById("sim-fixed-strain-hrs").innerText = `${f.hours_in_high_driving_pressure} Hours`;

  document.getElementById("sim-adapt-min-spo2").innerText = `${a.min_spo2}%`;
  document.getElementById("sim-adapt-max-dp").innerText = `${a.max_driving_pressure} cmH2O`;
  document.getElementById("sim-adapt-hypox-hrs").innerText = `${a.hours_in_hypoxemia} Hours`;
}

function updateSimSeverity(val) {
  const label = document.getElementById("sim-severity-val");
  if (val < 0.5) label.innerText = "Mild Lung Injury";
  else if (val < 0.8) label.innerText = "Moderate ARDS";
  else label.innerText = "Acute Severe ARDS";
}
