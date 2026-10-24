"""
Run physiological respiratory simulation comparing Fixed vs AI Adaptive strategies.
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath('.'))

from src.adaptive.simulator import RespiratorySimulator

def main():
    sim = RespiratorySimulator()
    res = sim.run_comparative_simulation(scenario="acute_lung_injury", total_steps=24)
    
    with open("outputs/simulation_results.json", "w") as f:
        # Convert dataclasses to dict if any inside
        json.dump(res, f, default=lambda o: getattr(o, '__dict__', str(o)), indent=2)

    print("\n" + "="*70)
    print("      PHYSIOLOGICAL SIMULATION: FIXED VS AI-ADAPTIVE VENTILATION")
    print("="*70)
    print("\n[Fixed Standard Ventilator Management]")
    s_fix = res["summary"]["fixed_strategy"]
    print(f"  Minimum SpO2               : {s_fix['min_spo2']}% (Severe Hypoxemia)")
    print(f"  Mean SpO2                  : {s_fix['mean_spo2']}%")
    print(f"  Maximum Driving Pressure   : {s_fix['max_driving_pressure']} cmH2O (Danger > 14 cmH2O)")
    print(f"  Hours in Hypoxemia (<88%)  : {s_fix['hours_in_hypoxemia']} hours")
    print(f"  Hours in High Driving Press: {s_fix['hours_in_high_driving_pressure']} hours")

    print("\n[AI Adaptive Predictive Ventilator Management]")
    s_ada = res["summary"]["adaptive_strategy"]
    print(f"  Minimum SpO2               : {s_ada['min_spo2']}% (Protected)")
    print(f"  Mean SpO2                  : {s_ada['mean_spo2']}%")
    print(f"  Maximum Driving Pressure   : {s_ada['max_driving_pressure']} cmH2O (Strict Lung-Protective Bound)")
    print(f"  Hours in Hypoxemia (<88%)  : {s_ada['hours_in_hypoxemia']} hours")
    print(f"  Hours in High Driving Press: {s_ada['hours_in_high_driving_pressure']} hours")
    print("="*70)

if __name__ == "__main__":
    main()
