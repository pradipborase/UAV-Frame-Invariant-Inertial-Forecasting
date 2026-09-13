# Sampling-rate compatibility (Stage 0C)

No resampling, interpolation, anti-alias filtering, or decimation was applied.

## Audited nominal/measured rates

- EuRoC documented: 200 Hz (ASL page; ADIS16448).
- EuRoC measured median (this audit): 200.003 Hz across audited sequences.
- UZH-FPV Snapdragon documented: 500 Hz.
- UZH-FPV measured (Stage 0, re-used): approximately 500 Hz.

## Candidate future common rates (not frozen)

### 100 Hz

- EuRoC 200 -> 100: integer ratio 2:1. Future work would still require an explicit anti-alias design before decimation.
- UZH 500 -> 100: integer ratio 5:1. Same anti-alias requirement.
- Information loss: both sources lose high-frequency inertial content; UZH loses more bandwidth relative to its native 500 Hz.
- Cross-dataset consistency: both can theoretically share 100 Hz after a later authorized stage.

### 200 Hz

- EuRoC 200 -> 200: identity (no rate conversion) if measured rate is acceptably 200 Hz class.
- UZH 500 -> 200: rational 5:2. Not integer decimation. Would require a documented resampling filter, not a naive stride.
- Information loss: EuRoC unchanged; UZH reduced.
- Cross-dataset consistency: possible but not a trivial integer downsample on UZH.

## What is not allowed yet

Implementing either conversion, estimating filter coefficients from target-test recordings, or claiming a frozen study rate.

Evidence: DERIVED_FROM_RAW_TIMESTAMPS (EuRoC this stage; UZH Stage 0).
