# Limitations fact sheet (severity order)

1. **n=6 EuRoC / n=8 UZH.** Why it matters: CIs are wide; majority votes are fragile. Mitigation: sequence unit, bootstrap CIs, wins/losses. Cannot claim: statistically superior models.

2. **Two datasets / two platforms.** Why: transfer is not a universal UAV law. Mitigation: official heterogeneous pair. Cannot claim: generalization to all UAV platforms.

3. **Magnitude target loses direction; UZH body frame unresolved.** Why: not a 3-axis body-frame estimator. Mitigation: Stage 0C frame audit. Cannot claim: axis-aligned specific-force transfer.

4. **Target is future IMU observation, not independent acceleration truth.** Why: telemetry forecasting ≠ navigation reference. Mitigation: explicit estimand. Cannot claim: the method estimates true physical acceleration or uses ground-truth acceleration.

5. **No onboard deployment; CPU timing only.** Why: footprint ≠ real-time flight. Mitigation: wording restriction. Cannot claim: real-time onboard suitability.

6. **High-q_f intervals sparse; gravity near 9.81 m/s^2 is common.** Why: secondary regime / near-g sequences. Mitigation: pre-specified p95; gravity table. Cannot claim: extreme-maneuver certification.

7. **Datasets audited before modelling.** Why: not full blindness. Mitigation: no target-fitted objects. Cannot claim: the target was unseen throughout the study, or that the study is preregistered.

8. **No predictive uncertainty model.** Why: bootstrap is across flights. Cannot claim: GUM measurement uncertainty.
