"""Cd integration shared by preprocessing and model evaluation."""

from __future__ import annotations

import numpy as np

U_INF = 38.89  # DrivAerML freestream, m/s; surface WSS is kinematic.
Q_INF = 0.5 * U_INF**2


def integrate_cd(cp, cf_x, nx, area, a_ref):
    """Return pressure, friction and total Cd for outward body normals."""
    cp, cf_x, nx, area = map(np.asarray, (cp, cf_x, nx, area))
    if not (cp.shape == cf_x.shape == nx.shape == area.shape):
        raise ValueError("Surface arrays must align cell-for-cell")
    if a_ref <= 0 or np.any(area <= 0):
        raise ValueError("Reference and cell areas must be positive")
    pressure = float(np.sum(-cp * nx * area, dtype=np.float64) / a_ref)
    # DrivAerML's stored wallShearStressMeanTrim is wall-on-fluid traction.
    # The force on the vehicle is its negative; verified against force_mom Cd.
    friction = float(np.sum(-cf_x * area, dtype=np.float64) / a_ref)
    return {"pressure": pressure, "friction": friction, "total": pressure + friction}
