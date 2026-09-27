"""Joint-angle geometry used to annotate tracked surgeon pose landmarks."""

import numpy as np


def calculate_angle(a, b, c) -> float:
    """Angle at vertex `b`, given three 2D landmark points (a, b, c), in degrees.

    Used here for shoulder, elbow and wrist angles, but works for any
    three-point joint angle from MediaPipe Pose landmarks.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 2)


# (hip, shoulder, elbow), (shoulder, elbow, wrist), (elbow, wrist, index) per side.
TRACKED_JOINTS = {
    "shoulder": ("hip", "shoulder", "elbow"),
    "elbow": ("shoulder", "elbow", "wrist"),
    "wrist": ("elbow", "wrist", "index"),
}
