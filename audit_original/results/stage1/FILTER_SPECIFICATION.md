# Causal anti-alias filter specification (Stage 1)

No prediction performance was used to choose these edges.

## Locked physical requirements (both datasets)

- Filter class: causal IIR, elliptic (`ellip`), second-order sections (SOS)
- Implementation: `scipy.signal.sosfilt` forward only
- Forbidden: `filtfilt`, `sosfiltfilt`, zero-phase, centred moving averages
- Passband edge: 30.0 Hz
- Stopband edge: 45.0 Hz
- Passband ripple: <= 1.0 dB
- Stopband attenuation: >= 60.0 dB
- FIT_SCOPE: NONE_DETERMINISTIC_PHYSICAL

## Native-rate choice

Documented nominal rates are used for filter design and integer decimation:

- EuRoC: 200 Hz (Stage 0C measured median 200.00256 Hz). Integer factor 2 -> 100 Hz.
- UZH-FPV: 500 Hz (Stage 0 measured Snapdragon ~500 Hz). Integer factor 5 -> 100 Hz.

The 0.00256 Hz EuRoC offset is sampling jitter, not a different Nyquist grid. Designing at 200.00256 Hz would break integer decimation. Nominal documented rates are therefore the scientifically appropriate design frequencies.

## EuRoC design (fs = 200 Hz)

- type: ellip
- order: 6 (3 SOS sections)
- max pole radius: 0.943485417547
- stable: True
- causal: YES
- maximum passband loss: 1 dB
- minimum stopband attenuation: 60 dB
- passband group-delay median (samples): 4.13107
- passband group-delay max (samples): 19.4352
- impulse-response relative tail after 1.0 s: 4.00743e-06
- SOS coefficients (b0,b1,b2,a0,a1,a2 per section):
```
[[ 0.0091318170074144  0.005350949480373   0.0091318170074144
   1.                 -0.7154723854883437  0.                ]
 [ 1.                  1.                  0.
   1.                 -1.2780591166411965  0.65515191910096  ]
 [ 1.                 -0.2390030141857213  1.
   1.                 -1.1149960257436253  0.890164733124431 ]]
```

## UZH design (fs = 500 Hz)

- type: ellip
- order: 6 (3 SOS sections)
- max pole radius: 0.985014450226
- stable: True
- causal: YES
- maximum passband loss: 1 dB
- minimum stopband attenuation: 60 dB
- passband group-delay median (samples): 12.999
- passband group-delay max (samples): 74.3579
- impulse-response relative tail after 1.0 s: 0.00015431
- SOS coefficients:
```
[[ 1.5855758865265046e-03 -5.8721120498604075e-04  1.5855758865265042e-03
   1.0000000000000000e+00 -1.7896232979513227e+00  8.1342355958889545e-01]
 [ 1.0000000000000000e+00 -1.6050513520512639e+00  9.9999999999999967e-01
   1.0000000000000000e+00 -1.8055888662122941e+00  8.9544045541022332e-01]
 [ 1.0000000000000000e+00 -1.7425815108590852e+00  9.9999999999999989e-01
   1.0000000000000000e+00 -1.8324197734320145e+00  9.7025346715314031e-01]]
```

## Startup transient / warmup

Causal IIR filters have a startup transient. It is **not** compensated with future samples.

A conservative **1.0 s** physical-time exclusion is applied to both datasets after filtering and decimation.

Reason: at 1.0 s the relative impulse-response tail is ~4e-6 (200 Hz design) and ~1.5e-4 (500 Hz design), far below the 1 s physical-time rule requested for a shared warmup. The same physical duration is used on both datasets.

## Group delay

Group delay is frequency-dependent (elliptic). It is a deterministic property of the locked filter, not a fitted lag. No zero-phase inversion is applied because that would leak the future.
