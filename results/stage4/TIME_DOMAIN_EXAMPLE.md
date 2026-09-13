# Time-domain example figure (optional; not a primary figure)

Pre-defined selection rule (neutral, not cherry-picked):
within each dataset, choose the sequence whose Ridge multi-horizon RMSE is the median among sequences in that dataset.

Status: **not generated**.

Reason: Stage 2 and Stage 3 did not save per-window prediction traces. Regenerating traces would require re-running frozen models. Stage 4 is forbidden from training or from regenerating forecasts except to verify an already frozen numeric result. Loading already-saved window-level predictions is allowed; those files are absent.

If a later reproducibility audit restores frozen window-level predictions without refitting, the median-Ridge sequence rule above remains the only authorized selection rule.
