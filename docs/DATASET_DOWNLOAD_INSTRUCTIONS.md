# Dataset download and placement

Raw **EuRoC MAV** and **UZH-FPV** files are **not** in this package. Download them from the official hosts only. Do not use Kaggle, AcademicTorrents, or unofficial preprocessed dumps.

This study uses **14 recordings**: 6 EuRoC Vicon sequences and 8 UZH-FPV Snapdragon sequences.

Treat this `GITHUB_RELEASE_V3` folder as the project root (`ROOT`) when placing files.

## Exact recordings

### EuRoC MAV (D1)

Official landing page: https://projects.asl.ethz.ch/datasets/euroc-mav/  
ETH Research Collection (DOI): https://doi.org/10.3929/ethz-b-000690084  
Licence (dataset): In Copyright – Non-Commercial Use Permitted (InC-NC 1.0)

| Sequence ID | Bundle |
| --- | --- |
| `V1_01_easy` | `vicon_room1.zip` |
| `V1_02_medium` | `vicon_room1.zip` |
| `V1_03_difficult` | `vicon_room1.zip` |
| `V2_01_easy` | `vicon_room2.zip` |
| `V2_02_medium` | `vicon_room2.zip` |
| `V2_03_difficult` | `vicon_room2.zip` |

Do **not** download Machine Hall sequences for this paper. They were not used.

### UZH-FPV Snapdragon (D2)

Official pages: https://fpv.ifi.uzh.ch/datasets/  
v3 text archives (prefer v3; official 30 Mar 2020 GT time-offset correction):  
`http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/`  
Licence (dataset): CC BY-NC-SA 3.0

| Sequence ID | Official zip name |
| --- | --- |
| `indoor_forward_6_snapdragon` | `indoor_forward_6_snapdragon_with_gt.zip` |
| `indoor_forward_9_snapdragon` | `indoor_forward_9_snapdragon_with_gt.zip` |
| `indoor_forward_10_snapdragon` | `indoor_forward_10_snapdragon_with_gt.zip` |
| `indoor_45_2_snapdragon` | `indoor_45_2_snapdragon_with_gt.zip` |
| `indoor_45_4_snapdragon` | `indoor_45_4_snapdragon_with_gt.zip` |
| `indoor_45_13_snapdragon` | `indoor_45_13_snapdragon_with_gt.zip` |
| `indoor_45_14_snapdragon` | `indoor_45_14_snapdragon_with_gt.zip` |
| `outdoor_forward_1_snapdragon` | `outdoor_forward_1_snapdragon_with_gt.zip` |

Use the **Snapdragon** `*_with_gt.zip` text archives. DAVIS IMU streams are a different sensor and are not interchangeable in this protocol.

## Where to put files

### EuRoC raw bundles

```text
ROOT/data/raw/euroc/vicon_room1.zip
ROOT/data/raw/euroc/vicon_room2.zip
```

Official bitstream URLs used in this study (ETH Research Collection API):

- `vicon_room1.zip`: `https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content`
- `vicon_room2.zip`: `https://www.research-collection.ethz.ch/server/api/core/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content`

Helper (optional):

```text
python scripts/download_stage0c_data.py
python scripts/run_stage0c_audit.py
```

`run_stage0c_audit.py` extracts **text IMU/YAML members only** (not images or bags) into:

```text
ROOT/data/raw/euroc/extracted/{SEQUENCE}.zip
ROOT/data/cache/euroc/{SEQUENCE}/mav0/imu0/data.csv
```

Stage 1 reads EuRoC IMU from that cache path.

### UZH-FPV raw zips

```text
ROOT/data/raw/uzh_fpv/sequences/indoor_forward_6_snapdragon_with_gt.zip
ROOT/data/raw/uzh_fpv/sequences/indoor_forward_9_snapdragon_with_gt.zip
ROOT/data/raw/uzh_fpv/sequences/indoor_forward_10_snapdragon_with_gt.zip
ROOT/data/raw/uzh_fpv/sequences/indoor_45_2_snapdragon_with_gt.zip
ROOT/data/raw/uzh_fpv/sequences/indoor_45_4_snapdragon_with_gt.zip
ROOT/data/raw/uzh_fpv/sequences/indoor_45_13_snapdragon_with_gt.zip
ROOT/data/raw/uzh_fpv/sequences/indoor_45_14_snapdragon_with_gt.zip
ROOT/data/raw/uzh_fpv/sequences/outdoor_forward_1_snapdragon_with_gt.zip
```

Example URL:

```text
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_6_snapdragon_with_gt.zip
```

Stage 1 reads UZH IMU from the extracted text file:

```text
ROOT/data/cache/uzh_fpv/sequences/{SEQUENCE}_with_gt/imu.txt
```

`python scripts/run_stage0c_audit.py` extracts `imu.txt` from the official zip if the cache copy is missing.

## After placement

Do not edit raw archives. Preprocessing writes only to `data/processed/stage1/` (created at runtime; not shipped here).

Continue with `docs/REPRODUCTION_INSTRUCTIONS.md`.
