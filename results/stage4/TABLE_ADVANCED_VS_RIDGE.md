# Table 4. Advanced versus Ridge (paired sequence differences)

Delta = RMSE_advanced − RMSE_Ridge (m/s^2). Negative Delta means the advanced model has lower sequence RMSE.

| Context | Model | Mean Δ | Median Δ | 95% CI | Wins–Losses–Ties | Relative % vs Ridge | Evidence class |
|---|---|---|---|---|---|---|---|
| WITHIN_EUROC | TCN | 0.007 | 0.009 | [-0.005, 0.016] | 1–5–0 | -0.9 | C. NO MATERIAL GAIN |
| WITHIN_EUROC | GRU | 0.003 | 0.010 | [-0.010, 0.014] | 2–4–0 | -0.5 | C. NO MATERIAL GAIN |
| WITHIN_EUROC | Transformer | 0.016 | 0.024 | [-0.005, 0.032] | 1–5–0 | -2.2 | D. DEGRADATION |
| WITHIN_UZH | TCN | -0.023 | -0.020 | [-0.047, 0.003] | 6–2–0 | 1.2 | B. MODEST / UNCERTAIN GAIN |
| WITHIN_UZH | GRU | 0.003 | -0.012 | [-0.039, 0.049] | 5–3–0 | -0.1 | C. NO MATERIAL GAIN |
| WITHIN_UZH | Transformer | -0.048 | -0.040 | [-0.073, -0.022] | 7–1–0 | 2.5 | A. CLEAR PRACTICAL GAIN |
| EUROC_TO_UZH | TCN | -0.015 | -0.007 | [-0.044, 0.011] | 6–2–0 | 0.7 | B. MODEST / UNCERTAIN GAIN |
| EUROC_TO_UZH | GRU | 0.015 | -0.011 | [-0.045, 0.082] | 4–4–0 | -0.7 | C. NO MATERIAL GAIN |
| EUROC_TO_UZH | Transformer | 0.132 | 0.096 | [0.023, 0.254] | 1–7–0 | -6.4 | D. DEGRADATION |
| UZH_TO_EUROC | TCN | 0.010 | -0.004 | [-0.010, 0.037] | 3–3–0 | -1.2 | C. NO MATERIAL GAIN |
| UZH_TO_EUROC | GRU | -0.011 | -0.020 | [-0.033, 0.012] | 4–2–0 | 1.3 | B. MODEST / UNCERTAIN GAIN |
| UZH_TO_EUROC | Transformer | 0.002 | 0.004 | [-0.020, 0.019] | 2–4–0 | -0.2 | C. NO MATERIAL GAIN |
