# EuRoC source verification (Stage 0C)

retrieval_utc: 2026-09-12T16:56:32Z

## Dataset official name
The EuRoC MAV Dataset / EuRoC micro aerial vehicle datasets. Evidence: VERIFIED_FROM_OFFICIAL_WEB (ETH ASL page; ETH Research Collection item).

## Institution
ETH Zurich Autonomous Systems Lab (ASL). Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Dataset DOI
10.3929/ethz-b-000690084 (handle 20.500.11850/690084; item UUID bcaf173e-5dac-484b-bc37-faf97a594f1f). Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Publication DOI
10.1177/0278364915620033 — Burri et al., International Journal of Robotics Research, "The EuRoC micro aerial vehicle datasets". Evidence: VERIFIED_FROM_DATASET_PAPER; VERIFIED_FROM_OFFICIAL_WEB.

## Official landing page
ASL: https://projects.asl.ethz.ch/datasets/euroc-mav/
ETH Research Collection: https://www.research-collection.ethz.ch/handle/20.500.11850/690084
Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Download host
ETH Research Collection DSpace bitstream API: https://www.research-collection.ethz.ch/server/api/core/bitstreams
Evidence: VERIFIED_FROM_OFFICIAL_WEB (item JSON `_links.content.href`).

## License/rights
In Copyright – Non-Commercial Use Permitted (InC-NC 1.0)
Rights URL: http://rightsstatements.org/page/InC-NC/1.0/
ETH Library deposit terms also present as `license.txt` on the item.
Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Available archives (ORIGINAL bundle)
| archive | size_bytes | uuid | Stage 0C action |
|---|---:|---|---|
| vicon_room1.zip | 6042263426 | 02ecda9a-298f-498b-970c-b7c44334d880 | PHASE A download |
| vicon_room2.zip | 6013384949 | ea12bc01-3677-4b4c-853d-87c7870b8c44 | PHASE A download |
| machine_hall.zip | 12683729426 | 7b2419c1-62b5-4714-b7f8-485e5fe3e5fe | OPTIONAL / not downloaded (would exceed remaining budget with Vicon) |
| calibration_datasets.zip | 4416030888 | 5732e864-10f1-49e7-befb-669ee29ff770 | NOT REQUIRED if sequence YAML present |
| euroc_mav_dataset.pdf | 368358 | d861e63b-cfa9-4411-85a5-5ad6b3526e44 | metadata downloaded |

Vicon Phase A total = 12055648375 bytes (11.228 GiB) < 15 GiB additional budget (16106127360 bytes). Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Official documentation references
- ETH ASL EuRoC page (sensor list, known issues, sequence names)
- Burri et al. IJRR 2016
- ETH ASL `dataset_tools` (https://github.com/ethz-asl/dataset_tools) MATLAB loader `dataset_load_sensor_data.m`
- ASL Dataset Format YAML (`sensor.yaml`, `T_BS`)
Evidence: VERIFIED_FROM_OFFICIAL_WEB; VERIFIED_FROM_OFFICIAL_SOURCE_CODE.

## Sequence organization
Official zip.json previews list nested `{sequence}/{sequence}.zip` and `{sequence}/{sequence}.bag` inside room bundles. Evidence: VERIFIED_FROM_OFFICIAL_WEB (bitstream JSON previews under `data/metadata/euroc/`).

Standard flight recordings documented on ASL page and confirmed in zip.json:
V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult,
MH_01_easy, MH_02_easy, MH_03_medium, MH_04_difficult, MH_05_difficult.
Evidence: VERIFIED_FROM_OFFICIAL_WEB. Names were not assumed from memory.

## Known dataset limitations (official)
- Independent auto-exposure on the two cameras (stereo brightness mismatch; mid-exposure times aligned).
- Highly dynamic motion can degrade laser-tracker accuracy (Machine Hall).
- Sensor vs motion-capture recorded on different systems; Vicon device timestamps unavailable; temporal offset estimated.
Evidence: VERIFIED_FROM_OFFICIAL_WEB (ASL known-issues text in `euroc_mav_dataset.pdf.txt`).

## Sources not used
Kaggle, Google Drive mirrors, random GitHub dataset copies, preprocessed ML repositories. Evidence: this Stage 0C configuration.

## Official parser IMU layout (before file headers)
`dataset_load_sensor_data.m` case `imu`: uint64 timestamp, then 3 omega, then 3 accelerometer values. Plot labels: accelerometer `[m / s / s]`, gyro converted from rad to deg. Timestamps divided by 1e9. Evidence: VERIFIED_FROM_OFFICIAL_SOURCE_CODE. Exact CSV header strings are still taken from downloaded files (VERIFIED_FROM_RAW_FILE).
