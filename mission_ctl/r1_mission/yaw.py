"""Heading from the bridge's IMU quaternion.

The bridge already publishes everything needed on ~/imu at 50 Hz:

    [qw, qx, qy, qz, gx, gy, gz, gravx, gravy, gravz]

so closed-loop turning needs no bridge change at all. Component order is
(w, x, y, z) -- confirmed by measurement in W06, and getting it wrong yields a
plausible-looking angle that is simply not the robot's heading.

Limit worth stating plainly: this is a 6-axis IMU with no magnetometer, so yaw is
gyro-integrated and drifts without bound. Over a turn lasting a few seconds that
is negligible; over minutes it is not, which is why yaw is used to END a turn and
never to hold a heading across a long walk.
"""
from __future__ import annotations

import math


def quat_to_yaw(q):
    """(w, x, y, z) -> yaw in radians, in (-pi, pi]. Valid at any pitch/roll."""
    w, x, y, z = (float(v) for v in q[:4])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class YawTracker(object):
    """Unwraps yaw into a continuous accumulated angle.

    Without unwrapping, a turn that crosses +-pi reads as a 360 deg jump and a
    closed-loop turn either stops immediately or never stops.
    """

    def __init__(self):
        self._last = None
        self.total = 0.0     # accumulated, continuous, radians
        self.samples = 0

    def update(self, q):
        y = quat_to_yaw(q)
        if self._last is None:
            self._last = y
            self.samples = 1
            return self.total
        d = y - self._last
        # Shortest way round: a real step between two 50 Hz samples is tiny.
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        self.total += d
        self._last = y
        self.samples += 1
        return self.total

    @property
    def ready(self):
        return self.samples > 0
