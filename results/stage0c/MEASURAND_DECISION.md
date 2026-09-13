# Measurand decision (Stage 0C)

Physical sensor quantity and future prediction-error metric are distinct.
Example: the sensor indication is accelerometer specific force in m/s^2; a later RMSE would be an error of that indication, not a new physical quantity.

## Candidate A — 3-axis specific force, native IMU frame

- Definition: vector f = (fx, fy, fz) of accelerometer specific force in the dataset's IMU sensor frame.
- Unit: m/s^2
- Frame: EuRoC S via a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]; UZH S via lin_acc_x/y/z
- Required transforms: none if native frames are retained
- Traceability: headers + official unit documentation; gravity remains in the indication
- Advantages: retains directional inertial information; no target-fitted alignment
- Disadvantages: EuRoC S and UZH S are not the same physical axes; 3-axis errors are not axis-wise comparable without a documented common frame
- Measurement-journal relevance: honest sensor indication; must not be mislabelled linear acceleration

## Candidate B — 3-axis specific force in a common body frame

- Definition: R_fixed f_S mapped into a justified vehicle body
- EuRoC: T_BS exists (S->B) from sensor.yaml
- UZH: vehicle-body CAD UNRESOLVED; only camera-IMU Kalibr is documented
- Required transforms: would need an official UZH body convention that does not exist in the public audit
- Target-fitted rotation: prohibited
- Decision: not currently definable without inventing a UZH body

## Candidate C — specific-force magnitude ||f||

- Definition: sqrt(fx^2+fy^2+fz^2)
- Advantages: invariant under proper orthonormal frame rotations; avoids arbitrary axis pairing
- Limitations: loses direction; biases/noise remain; magnitude distributions differ across flight regimes
- Must not be selected merely because it is easier to compare

## Candidate D — 3-axis angular velocity

- EuRoC w_RS_S [rad/s]; UZH ang_vel_* [rad/s]
- Same frame caveat as Candidate A

## Candidate E — angular-speed magnitude ||omega||

- Rotation-invariant analog of Candidate C, with the same loss of direction

## Physical quantity vs evaluation quantity

Do not confuse the measurand with a future scoring rule.

## Recommendation

RECOMMENDED_COMMON_MEASURAND = 3-axis accelerometer specific force in each dataset's native IMU sensor frame (EuRoC a_RS_S; UZH lin_acc_*), SI unit m/s^2

confidence = HIGH

Rationale: both official IMU streams, after file audit, are accelerometer specific force in SI units with gravity remaining. A shared vehicle body is not officially available for UZH. Native-frame 3-axis specific force is therefore the scientifically richest quantity that can be defined without target-test fitting. Magnitude is a documented optional invariant, not the primary selection.

Losing directionality is not required for the intended inertial-measurement study and is not treated as acceptable merely to manufacture axis alignment.
