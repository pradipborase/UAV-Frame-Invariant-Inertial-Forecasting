"""Official-metadata sequence catalogues for Blackbird and UZH-FPV.

These tables are compiled from official README/dataset pages and official
accompanying code. They are not inferred from third-party papers' preprocessed dumps.
"""

from __future__ import annotations

from typing import Any

BLACKBIRD_FOLDER = {
    "3D Figure 8": "3dFigure8",
    "Ampersand": "ampersand",
    "Bent Dice": "bentDice",
    "Clover": "clover",
    "Dice": "dice",
    "Flat Figure 8": "flatFigure8",
    "Half-Moon": "halfMoon",
    "Mouse": "mouse",
    "Oval": "oval",
    "Patrick": "patrick",
    "Picasso": "picasso",
    "Sid": "sid",
    "Sphinx": "sphinx",
    "Star": "star",
    "Thrice": "thrice",
    "Tilted Thrice": "tiltedThrice",
    "Winter": "winter",
}

# Folder names for 3D/Flat Figure 8 are not present in sequenceDownloader examples.
# They appear in the official README tables as calibration flights. Evidence:
# VERIFIED_FROM_OFFICIAL_DOCUMENTATION for existence; folder slug UNRESOLVED.
BLACKBIRD_FOLDER_EVIDENCE = {
    "3dFigure8": "UNRESOLVED_FOLDER_SLUG",
    "flatFigure8": "UNRESOLVED_FOLDER_SLUG",
}

# Speeds marked present in the official GitHub README trajectory tables.
BLACKBIRD_YAW_FORWARD: dict[str, list[float]] = {
    "Ampersand": [1.0, 2.0],
    "Bent Dice": [0.5, 1.0, 2.0, 3.0],
    "Clover": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    "Dice": [1.0, 2.0, 3.0],
    "Half-Moon": [1.0, 2.0, 3.0, 4.0],
    "Mouse": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "Oval": [1.0, 2.0, 3.0, 4.0],
    "Patrick": [0.5, 1.0, 2.0, 3.0, 4.0],
    "Picasso": [0.5, 1.0, 3.0, 4.0, 5.0],
    "Sid": [0.5, 1.0, 2.0, 3.0, 4.0],
    "Sphinx": [1.0, 2.0, 3.0, 4.0],
    "Star": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    "Thrice": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "Tilted Thrice": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "Winter": [0.5, 2.0, 3.0, 4.0],
}

BLACKBIRD_YAW_CONSTANT: dict[str, list[float]] = {
    "3D Figure 8": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    "Ampersand": [1.0, 2.0, 3.0],
    "Bent Dice": [1.0, 2.0, 3.0, 4.0],
    "Clover": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "Dice": [2.0, 3.0, 4.0],
    "Flat Figure 8": [0.5, 1.0, 2.0, 3.0, 5.0],
    "Half-Moon": [1.0, 2.0, 3.0, 4.0],
    "Mouse": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "Oval": [2.0, 3.0, 4.0],
    "Patrick": [1.0, 2.0, 3.0, 4.0, 5.0],
    "Picasso": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "Sid": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "Sphinx": [1.0, 2.0, 3.0, 4.0],
    "Star": [1.0, 2.0, 3.0, 4.0, 5.0],
    "Thrice": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "Tilted Thrice": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "Winter": [1.0, 2.0, 3.0, 4.0, 5.0],
}

BLACKBIRD_CALIBRATION_NO_CAMERA = {
    ("3D Figure 8", "yawConstant"),
    ("Flat Figure 8", "yawConstant"),
}

BLACKBIRD_CSV_FILES = [
    "blackbird_slash_imu.csv",
    "blackbird_slash_pose_ref.csv",
    "blackbird_slash_pwm.csv",
    "blackbird_slash_rotor_rpm.csv",
    "blackbird_slash_state.csv",
    "camera_d_slash_camera_info.csv",
    "camera_l_slash_camera_info.csv",
    "camera_r_slash_camera_info.csv",
    "tf.csv",
]

BLACKBIRD_FLIGHT_FILES = [
    "rosbag.bag",
    "flightNormalizationOffset.csv",
    "groundTruthPoses.csv",
]


def speed_bucket(speed_mps: float) -> str:
    if speed_mps <= 2.0:
        return "low"
    if speed_mps <= 4.0:
        return "medium"
    return "high"


def _speed_token(speed: float) -> str:
    whole = int(speed)
    frac = int(round((speed - whole) * 10))
    return f"maxSpeed{whole}p{frac}"


def blackbird_sequences() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for yaw, table in (("yawForward", BLACKBIRD_YAW_FORWARD), ("yawConstant", BLACKBIRD_YAW_CONSTANT)):
        for family, speeds in table.items():
            folder = BLACKBIRD_FOLDER[family]
            for speed in speeds:
                token = _speed_token(speed)
                sequence_id = f"{folder}/{yaw}/{token}"
                calib_only = (family, yaw) in BLACKBIRD_CALIBRATION_NO_CAMERA
                rows.append(
                    {
                        "dataset": "BLACKBIRD",
                        "sequence_id": sequence_id,
                        "recording_id": sequence_id,
                        "platform": "Blackbird_Xsens_MTi3_quadrotor",
                        "sensor_platform": "Xsens_MTi-3",
                        "trajectory_name": family,
                        "trajectory_family": folder,
                        "yaw_condition": yaw,
                        "yaw_mode": yaw,
                        "speed_condition": speed_bucket(speed),
                        "nominal_max_speed_mps": speed,
                        "environment": "motion_capture_room_physical__visual_environment_is_render_only",
                        "environment_if_applicable": "UNRESOLVED_UNTIL_RENDER_SELECTED",
                        "recording_or_flight_identifier": sequence_id,
                        "imu_available": "YES_DOCUMENTED",
                        "ground_truth_availability": "YES_DOCUMENTED_POSE",
                        "motor_data_availability": "YES_DOCUMENTED",
                        "actual_sensor_archive_availability": "OFFICIAL_SERVER_REQUIRED",
                        "camera_archive_availability": "NO" if calib_only else "OPTIONAL_RENDER_NOT_REQUIRED_FOR_STAGE0",
                        "download_availability": "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE",
                        "official_source_evidence": "VERIFIED_FROM_OFFICIAL_DOCUMENTATION",
                        "same_trajectory_group": folder,
                        "same_environment_group": "blackbird_mocap_volume",
                        "potential_dependency_group": folder,
                        "independent_recording_evidence": (
                            "Official papers state each flight is a distinct recording of a "
                            "nominal trajectory at a chosen speed/yaw. Different environments "
                            "are FlightGoggles renders of the same physical flight and are NOT "
                            "independent IMU/GT recordings."
                        ),
                        "independence_confidence": "HIGH" if True else "HIGH",
                        "calibration_flight_no_camera": calib_only,
                        "folder_slug_evidence": BLACKBIRD_FOLDER_EVIDENCE.get(folder, "VERIFIED_FROM_OFFICIAL_DOCUMENTATION"),
                    }
                )
    return rows


UZH_SEQUENCES: list[dict[str, Any]] = [
    # Indoor forward
    {"nr": 3, "env": "indoor", "cam": "forward", "duration_s": 54.63, "length_m": 287.12, "vmax": 9.5, "davis_bag": "853.7MB", "davis_zip": "413.9MB", "snap_bag": "1.6GB", "snap_zip": "842.8MB", "leica": "76kB", "public_gt": True, "difficulty": "Easy"},
    {"nr": 5, "env": "indoor", "cam": "forward", "duration_s": 50.0, "length_m": 156.47, "vmax": 4.87, "davis_bag": "1.3GB", "davis_zip": "655.3MB", "snap_bag": "2.6GB", "snap_zip": "1.3GB", "leica": "142kB", "public_gt": True, "difficulty": "Medium"},
    {"nr": 6, "env": "indoor", "cam": "forward", "duration_s": 32.93, "length_m": 223.27, "vmax": 12.52, "davis_bag": "728.1MB", "davis_zip": "365.1MB", "snap_bag": "1.2GB", "snap_zip": "605.3MB", "leica": "58kB", "public_gt": True, "difficulty": "Medium"},
    {"nr": 7, "env": "indoor", "cam": "forward", "duration_s": 73.2, "length_m": 333.59, "vmax": 12.78, "davis_bag": "1.1GB", "davis_zip": "559.5MB", "snap_bag": "2GB", "snap_zip": "964.1MB", "leica": "95kB", "public_gt": True, "difficulty": "Hard"},
    {"nr": 8, "env": "indoor", "cam": "forward", "duration_s": 132.53, "length_m": 259.16, "vmax": 5.26, "davis_bag": "1.4GB", "davis_zip": "726.9MB", "snap_bag": "2.8GB", "snap_zip": "1.4GB", "leica": None, "public_gt": False, "difficulty": None},
    {"nr": 9, "env": "indoor", "cam": "forward", "duration_s": 34.04, "length_m": 157.07, "vmax": 11.42, "davis_bag": "710.5MB", "davis_zip": "365.6MB", "snap_bag": "1.3GB", "snap_zip": "627MB", "leica": "68kB", "public_gt": True, "difficulty": "Easy"},
    {"nr": 10, "env": "indoor", "cam": "forward", "duration_s": 33.43, "length_m": 149.36, "vmax": 9.49, "davis_bag": "643.7MB", "davis_zip": "327.9MB", "snap_bag": "1.3GB", "snap_zip": "678.7MB", "leica": "69kB", "public_gt": True, "difficulty": "Easy"},
    {"nr": 11, "env": "indoor", "cam": "forward", "duration_s": 24.02, "length_m": 85.68, "vmax": 10.32, "davis_bag": "973.2MB", "davis_zip": "467.5MB", "snap_bag": "1.4GB", "snap_zip": "731.3MB", "leica": None, "public_gt": False, "difficulty": None},
    {"nr": 12, "env": "indoor", "cam": "forward", "duration_s": 31.98, "length_m": 124.07, "vmax": 15.28, "davis_bag": "565.1MB", "davis_zip": "268.4MB", "snap_bag": "1GB", "snap_zip": "567.8MB", "leica": None, "public_gt": False, "difficulty": None},
    # Indoor 45
    {"nr": 1, "env": "indoor", "cam": "45", "duration_s": 73.99, "length_m": 150.71, "vmax": 4.36, "davis_bag": "1.3GB", "davis_zip": "635.8MB", "snap_bag": "1.6GB", "snap_zip": "897.3MB", "leica": None, "public_gt": False, "difficulty": None},
    {"nr": 2, "env": "indoor", "cam": "45", "duration_s": 55.77, "length_m": 218.9, "vmax": 6.97, "davis_bag": "1.3GB", "davis_zip": "602.5MB", "snap_bag": "1.2GB", "snap_zip": "643.5MB", "leica": "83kB", "public_gt": True, "difficulty": "Easy"},
    {"nr": 3, "env": "indoor", "cam": "45", "duration_s": 57.82, "length_m": 119.82, "vmax": 3.53, "davis_bag": "870.6MB", "davis_zip": "437.8MB", "snap_bag": "1.3GB", "snap_zip": "683.1MB", "leica": None, "public_gt": False, "difficulty": None},
    {"nr": 4, "env": "indoor", "cam": "45", "duration_s": 47.36, "length_m": 168.06, "vmax": 6.55, "davis_bag": "1.1GB", "davis_zip": "496.3MB", "snap_bag": "1.1GB", "snap_zip": "607.8MB", "leica": "71kB", "public_gt": True, "difficulty": "Easy"},
    {"nr": 9, "env": "indoor", "cam": "45", "duration_s": 40.0, "length_m": 215.58, "vmax": 11.23, "davis_bag": "808.1MB", "davis_zip": "370.8MB", "snap_bag": "1.2GB", "snap_zip": "664MB", "leica": "62kB", "public_gt": True, "difficulty": "Medium"},
    {"nr": 11, "env": "indoor", "cam": "45", "duration_s": 22.96, "length_m": 125.21, "vmax": 11.74, "davis_bag": "949.6MB", "davis_zip": "440.1MB", "snap_bag": "893.3MB", "snap_zip": "480.8MB", "leica": None, "public_gt": False, "difficulty": None},
    {"nr": 12, "env": "indoor", "cam": "45", "duration_s": 51.25, "length_m": 124.56, "vmax": 4.33, "davis_bag": "563.7MB", "davis_zip": "268.8MB", "snap_bag": "1.3GB", "snap_zip": "648.6MB", "leica": "72kB", "public_gt": True, "difficulty": "Easy"},
    {"nr": 13, "env": "indoor", "cam": "45", "duration_s": 42.49, "length_m": 166.62, "vmax": 7.92, "davis_bag": "527.9MB", "davis_zip": "254.7MB", "snap_bag": "1.1GB", "snap_zip": "534.9MB", "leica": "62kB", "public_gt": True, "difficulty": "Medium"},
    {"nr": 14, "env": "indoor", "cam": "45", "duration_s": 43.66, "length_m": 220.4, "vmax": 9.54, "davis_bag": "625.6MB", "davis_zip": "301MB", "snap_bag": "1.2GB", "snap_zip": "543.8MB", "leica": "60kB", "public_gt": True, "difficulty": "Hard"},
    {"nr": 16, "env": "indoor", "cam": "45", "duration_s": 15.49, "length_m": 58.72, "vmax": 7.69, "davis_bag": "336MB", "davis_zip": "166.1MB", "snap_bag": "805.3MB", "snap_zip": "448MB", "leica": None, "public_gt": False, "difficulty": None},
    # Outdoor forward
    {"nr": 1, "env": "outdoor", "cam": "forward", "duration_s": 49.63, "length_m": 258.23, "vmax": 8.55, "davis_bag": "687.2MB", "davis_zip": "358.6MB", "snap_bag": "1.5GB", "snap_zip": "664.8MB", "leica": "123kB", "public_gt": True, "difficulty": "Easy"},
    {"nr": 2, "env": "outdoor", "cam": "forward", "duration_s": 36.9, "length_m": 220.88, "vmax": 10.13, "davis_bag": "763.4MB", "davis_zip": "410.3MB", "snap_bag": "249.8MB", "snap_zip": "710.4MB", "leica": None, "public_gt": False, "difficulty": None},
    {"nr": 3, "env": "outdoor", "cam": "forward", "duration_s": 92.84, "length_m": 735.51, "vmax": 14.04, "davis_bag": "1.1GB", "davis_zip": "637.2MB", "snap_bag": "2.2GB", "snap_zip": "1.1GB", "leica": "117kB", "public_gt": True, "difficulty": "Medium"},
    {"nr": 5, "env": "outdoor", "cam": "forward", "duration_s": 22.21, "length_m": 189.63, "vmax": 20.73, "davis_bag": "657.1MB", "davis_zip": "342.7MB", "snap_bag": "1.5GB", "snap_zip": "729.6MB", "leica": "72kB", "public_gt": True, "difficulty": "Hard"},
    {"nr": 6, "env": "outdoor", "cam": "forward", "duration_s": 34.83, "length_m": 338.2, "vmax": 19.42, "davis_bag": "786.5MB", "davis_zip": "427MB", "snap_bag": "1.3GB", "snap_zip": "677.6MB", "leica": None, "public_gt": False, "difficulty": None},
    {"nr": 9, "env": "outdoor", "cam": "forward", "duration_s": 43.15, "length_m": 314.41, "vmax": 10.68, "davis_bag": "1.1GB", "davis_zip": "577.4MB", "snap_bag": "1.5GB", "snap_zip": "742.1MB", "leica": None, "public_gt": False, "difficulty": None},
    {"nr": 10, "env": "outdoor", "cam": "forward", "duration_s": 59.6, "length_m": 455.63, "vmax": 12.58, "davis_bag": "1.2GB", "davis_zip": "656MB", "snap_bag": "1.8GB", "snap_zip": "955.6MB", "leica": None, "public_gt": False, "difficulty": None},
    # Outdoor 45
    {"nr": 1, "env": "outdoor", "cam": "45", "duration_s": 24.49, "length_m": 165.53, "vmax": 15.62, "davis_bag": "1.3GB", "davis_zip": "650.4MB", "snap_bag": "1.4GB", "snap_zip": "723.3MB", "leica": "46kB", "public_gt": True, "difficulty": "Medium"},
    {"nr": 2, "env": "outdoor", "cam": "45", "duration_s": 26.19, "length_m": 143.13, "vmax": 10.68, "davis_bag": "1.1GB", "davis_zip": "588.1MB", "snap_bag": "1.2GB", "snap_zip": "633.5MB", "leica": None, "public_gt": False, "difficulty": None},
]


def uzh_speed_condition(vmax: float | None) -> str:
    if vmax is None:
        return "UNRESOLVED"
    if vmax < 6.0:
        return "low"
    if vmax <= 12.0:
        return "moderate"
    return "aggressive"


def uzh_cam_token(cam: str) -> str:
    return "forward" if cam == "forward" else "45"


def uzh_sequence_stem(row: dict[str, Any], sensor: str) -> str:
    return f"{row['env']}_{uzh_cam_token(row['cam'])}_{row['nr']}_{sensor}"


def uzh_sequences(sensor_platforms: tuple[str, ...] = ("snapdragon", "davis")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in UZH_SEQUENCES:
        for sensor in sensor_platforms:
            stem = uzh_sequence_stem(spec, sensor)
            public = bool(spec["public_gt"])
            zip_name = f"{stem}_with_gt.zip" if public else f"{stem}.zip"
            bag_name = f"{stem}_with_gt.bag" if public else f"{stem}.bag"
            rows.append(
                {
                    "dataset": "UZH_FPV",
                    "sequence_id": stem,
                    "recording_id": f"{spec['env']}_{uzh_cam_token(spec['cam'])}_{spec['nr']}",
                    "platform": "Lumenier_QAV-R_FPV_quadrotor",
                    "sensor_platform": sensor,
                    "environment": spec["env"],
                    "camera_orientation": "forward" if spec["cam"] == "forward" else "45_down",
                    "track_or_trajectory_category": f"{spec['env']}_{uzh_cam_token(spec['cam'])}",
                    "trajectory_family": f"{spec['env']}_{uzh_cam_token(spec['cam'])}",
                    "sequence_number": spec["nr"],
                    "yaw_condition": "UNRESOLVED_NOT_A_BLACKBIRD_YAW_MODE",
                    "speed_condition": uzh_speed_condition(spec["vmax"]),
                    "duration_s": spec["duration_s"],
                    "trajectory_length_m": spec["length_m"],
                    "maximum_speed_mps": spec["vmax"],
                    "snapdragon_availability": "YES_DOCUMENTED",
                    "davis_availability": "YES_DOCUMENTED",
                    "imu_available": "YES_DOCUMENTED",
                    "public_ground_truth": "YES" if public else "NO",
                    "public_gt": "YES" if public else "NO",
                    "ground_truth_source_type": (
                        "PUBLIC_BATCH_OPTIMIZED_POSE_FROM_LEICA_POSITION_AND_IMU"
                        if public
                        else "GROUND_TRUTH_WITHHELD"
                    ),
                    "calibration_availability": "YES_DOCUMENTED_KALIBR_YAML",
                    "text_archive_availability": "YES_OFFICIAL_ZIP",
                    "rosbag_availability": "YES_OFFICIAL_BAG",
                    "download_size": spec["snap_zip"] if sensor == "snapdragon" else spec["davis_zip"],
                    "benchmark_only_or_withheld_gt_status": "PUBLIC_GT" if public else "GROUND_TRUTH_WITHHELD",
                    "official_source_evidence": "VERIFIED_FROM_OFFICIAL_DOCUMENTATION",
                    "v3_zip_name": zip_name,
                    "v3_bag_name": bag_name,
                    "v3_url": f"http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/{zip_name}",
                    "leica_raw_url": (
                        f"http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/raw/"
                        f"{spec['env']}_{uzh_cam_token(spec['cam'])}_{spec['nr']}.zip"
                        if public
                        else None
                    ),
                    "same_trajectory_group": f"{spec['env']}_{uzh_cam_token(spec['cam'])}_{spec['nr']}",
                    "same_environment_group": spec["env"],
                    "potential_dependency_group": f"{spec['env']}_{uzh_cam_token(spec['cam'])}",
                    "independent_recording_evidence": (
                        "Official flags.py and dataset tables treat each numbered sequence as a "
                        "distinct flight. Snapdragon and DAVIS of the same number are concurrent "
                        "sensor streams from the same physical flight, not independent flights."
                    ),
                    "independence_confidence": "HIGH",
                    "difficulty_if_published": spec["difficulty"],
                    "documented_leica_size": spec["leica"],
                }
            )
    return rows


UZH_CALIB_FILES = [
    {
        "calib_id": "indoor_forward_calib_snapdragon",
        "url": "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/indoor_forward_calib_snapdragon.zip",
        "applies_to": "indoor_forward_*_snapdragon",
        "sensor_platform": "snapdragon",
    },
    {
        "calib_id": "indoor_45_calib_snapdragon",
        "url": "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/indoor_45_calib_snapdragon.zip",
        "applies_to": "indoor_45_*_snapdragon",
        "sensor_platform": "snapdragon",
    },
    {
        "calib_id": "outdoor_forward_calib_snapdragon",
        "url": "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/outdoor_forward_calib_snapdragon.zip",
        "applies_to": "outdoor_forward_*_snapdragon",
        "sensor_platform": "snapdragon",
    },
    {
        "calib_id": "outdoor_45_calib_snapdragon",
        "url": "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/outdoor_45_calib_snapdragon.zip",
        "applies_to": "outdoor_45_*_snapdragon",
        "sensor_platform": "snapdragon",
    },
]


SEQUENCE_INVENTORY_COLUMNS = [
    "dataset",
    "sequence_id",
    "recording_id",
    "platform",
    "sensor_platform",
    "environment",
    "trajectory_family",
    "yaw_condition",
    "speed_condition",
    "duration_s",
    "imu_available",
    "gt_available",
    "public_gt",
    "calibration_available",
    "same_trajectory_group",
    "same_environment_group",
    "potential_dependency_group",
    "independent_recording_evidence",
    "independence_confidence",
    "downloaded",
    "integrity_status",
    "usable_status",
    "exclusion_reason",
]
