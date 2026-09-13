# Stage 0C compatibility clarification

Do not rewrite historical Stage-0C files.

`STAGE0C_REPORT.md` reported:

- Critical findings = 0
- Major findings = 0

Those counts refer to **raw-data / integrity** checks in `EUROC_INTEGRITY_FINDINGS.csv`.
They are not a claim that cross-dataset compatibility limitations disappeared.

`EUROC_UZH_COMPATIBILITY.csv` contains **4** rows with severity=MAJOR:

- `IMU frame`: SAME_NAMING_DIFFERENT_PHYSICAL_AXES — keep native frames or use a documented invariant; never learn alignment on target test data
- `body transform`: NOT_A_SHARED_BODY_FRAME — do not invent a UZH vehicle body; do not fit a cross-dataset rotation
- `frame standardization feasibility`: NOT_FOR_A_SHARED_VEHICLE_BODY — do not learn a cross-dataset rotation; native frames or invariant scalars
- `3-axis specific-force feasibility`: SAME_QUANTITY_DIFFERENT_AXES — defensible if axes are not claimed identical

Counted exactly from the Stage-0C CSV (excluding the OVERALL_CLASSIFICATION row, which is INFO).

## Two different classes of 'MAJOR'

1. **RAW-DATA/INTEGRITY MAJOR FINDINGS** — file corruption, non-finite IMU, broken timestamps, missing GT. Stage 0C: none failed.
2. **CROSS-DATASET COMPATIBILITY MAJOR LIMITATIONS** — EuRoC and UZH IMU axes are not demonstrated to be the same vehicle directions; UZH vehicle-body CAD is unresolved; a shared body-axis 3-vector is not scientifically justified without a prohibited fitted rotation.

Stage 1 therefore adopts a **frame-invariant** primary representation (specific-force magnitude and angular-speed magnitude) specifically to avoid requiring an unjustified common body-axis mapping.

The compatibility limitations remain in force. They are managed by changing the measurand, not by declaring them absent.
