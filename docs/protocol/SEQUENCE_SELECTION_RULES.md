# SEQUENCE_SELECTION_RULES.md

Selection is frozen against forecasting performance. It uses official metadata only.

## Independent unit

An **actual recording** is the future inferential unit.

Not an overlapping window. Not a rendered camera environment of the same Blackbird flight. Not Snapdragon and DAVIS of the same UZH number treated as two flights.

## Blackbird hierarchy

`DATASET → PLATFORM → ENVIRONMENT (physical mocap volume) → TRAJECTORY FAMILY → SPEED CONDITION → YAW MODE → RECORDING`

Same family at nearby speeds: `potential_dependency_group` = trajectory family.

Rendered environments share IMU/GT: they are not extra recordings.

## UZH-FPV hierarchy

`DATASET → PLATFORM → ENVIRONMENT/CAMERA ORIENTATION → SEQUENCE NUMBER → SENSOR STREAM`

`recording_id` = environment + camera + number.  
`sequence_id` for Stage 0 Snapdragon = `recording_id` + `_snapdragon`.

Withheld GT: `GT_WITHHELD`. Never recovered unofficially.

## Stage-0 subset rules

- Blackbird: up to 12 flights, ≥4 families, ≥3 speed regimes if listed, more than one yaw mode, IMU+GT preferred, no images.
- UZH: up to 10 public-GT Snapdragon zips, indoor/outdoor and forward/45 coverage, mixed documented speeds.
- Minimum useful targets: 6 Blackbird and 6 UZH public-GT sequences. Missing Blackbird files → `BLACKBIRD_ACQUISITION_INCOMPLETE`, not fabricated replacements.

## Usability (quality only)

`USABLE_PRIMARY` normally requires official readable data, IMU accel+gyro, verified units, understood sensor frame, timestamps audited, public GT, ≥20 s IMU–GT overlap, ≥99% finite IMU, no corruption, documented calibration/frame relation.

Irregular sampling is reported, not an automatic exclusion.
