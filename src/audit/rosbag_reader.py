"""Optional ROS1 bag inspection without a ROS desktop install."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

BLACKBIRD_IMU_TOPIC = "/blackbird/imu"
BLACKBIRD_GT_TOPIC = "/blackbird/state"
UZH_SNAP_IMU_TOPIC = "/snappy_imu"
UZH_DAVIS_IMU_TOPIC = "/dvs/imu"
UZH_GT_ODOM_TOPIC = "/groundtruth/odometry"
UZH_GT_POSE_TOPIC = "/groundtruth/pose"


def bag_available() -> bool:
    try:
        import rosbags  # noqa: F401
        return True
    except Exception:
        return False


def summarize_bag(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "readable": False,
        "topics": [],
        "error": "",
    }
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
    except Exception as exc:
        summary["error"] = f"rosbags import failed: {exc}"
        return summary

    typestore = get_typestore(Stores.ROS1_COMMON)
    topics: list[dict[str, Any]] = []
    try:
        with Reader(path) as reader:
            for name, conn in reader.connections.items() if hasattr(reader, "connections") else []:
                topics.append(
                    {
                        "topic": getattr(conn, "topic", name),
                        "msgtype": getattr(conn, "msgtype", ""),
                        "msgcount": getattr(conn, "msgcount", None),
                    }
                )
            if not topics:
                for conn in reader.connections:
                    topics.append(
                        {
                            "topic": conn.topic,
                            "msgtype": conn.msgtype,
                            "msgcount": getattr(conn, "msgcount", None),
                        }
                    )
            summary["topics"] = topics
            summary["readable"] = True
            summary["typestore"] = "ROS1_COMMON"
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def read_imu_from_bag(path: Path, topic: str) -> dict[str, Any]:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_COMMON)
    stamps: list[float] = []
    acc: list[list[float]] = []
    gyro: list[list[float]] = []
    frame_ids: list[str] = []
    msgtype = ""
    with Reader(path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return {"error": f"topic not found: {topic}", "n": 0}
        msgtype = conns[0].msgtype
        for conn, timestamp, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(typestore.ros1_to_cdr(raw, conn.msgtype), conn.msgtype) if False else typestore.deserialize_ros1(raw, conn.msgtype)
            header = getattr(msg, "header", None)
            if header is not None:
                stamp = header.stamp
                stamps.append(float(stamp.sec) + float(stamp.nanosec if hasattr(stamp, "nanosec") else stamp.nsec) * 1e-9)
                frame_ids.append(str(header.frame_id))
            lin = msg.linear_acceleration
            ang = msg.angular_velocity
            acc.append([float(lin.x), float(lin.y), float(lin.z)])
            gyro.append([float(ang.x), float(ang.y), float(ang.z)])
    return {
        "topic": topic,
        "message_type": msgtype,
        "timestamps_s": np.asarray(stamps, dtype=np.float64),
        "accel": np.asarray(acc, dtype=np.float64),
        "gyro": np.asarray(gyro, dtype=np.float64),
        "frame_id": frame_ids[0] if frame_ids else "UNRESOLVED",
        "n": len(stamps),
        "accel_fields": {
            "x": "linear_acceleration.x",
            "y": "linear_acceleration.y",
            "z": "linear_acceleration.z",
        },
        "gyro_fields": {
            "x": "angular_velocity.x",
            "y": "angular_velocity.y",
            "z": "angular_velocity.z",
        },
        "timestamp_field": "header.stamp",
    }
