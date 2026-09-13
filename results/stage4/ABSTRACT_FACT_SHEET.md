# Abstract fact sheet (verified facts only; not abstract prose)

- Datasets: EuRoC MAV (n=6) and UZH-FPV Snapdragon (n=8); official sources only.
- Target: q_f = ||f||_2, accelerometer specific-force magnitude (m/s^2), 100 Hz.
- History/horizon: 1.0 s / 0.20 s; stride 0.05 s; report 50/100/200 ms.
- Models: Persistence; context-specific frozen Ridge; compact TCN, GRU, Transformer.
- Within EuRoC equal-sequence RMSE: Persistence 0.917; Ridge 0.728; TCN 0.735; GRU 0.732; Transformer 0.744 m/s^2.
- Within UZH: Persistence 2.349; Ridge 1.943; Transformer 1.895 m/s^2 (clearest advanced within-domain gain).
- EuRoC→UZH: Ridge 2.065; TCN 2.050; Transformer 2.198 m/s^2.
- UZH→EuRoC: Ridge 0.815; GRU 0.805 m/s^2.
- Main conclusion: compact nonlinear models can yield dataset-specific within-domain gains, but extra complexity does not consistently improve zero-shot transfer; Ridge remains competitive.
- Limitations: n=6/8; two datasets; magnitude loses direction; target is future sensor observation; no onboard deployment.
