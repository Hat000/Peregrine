"""Standard-atmosphere barometric altitude <-> pressure (ISA troposphere model).

The official sim exposes absolute pressure (HIGHRES_IMU.abs_pressure); the Elodin
rig exposes altitude directly. The estimator wants altitude. These two functions
are exact inverses, so a value round-trips cleanly through either representation,
and the same estimator code works against both sims (calibrate the reference at
arming, since only *relative* altitude matters here).
"""
from __future__ import annotations

_P0_HPA = 1013.25      # sea-level standard pressure (hPa)
_T0_K = 288.15         # sea-level standard temperature (K)
_L = 0.0065            # temperature lapse rate (K/m)
_G = 9.80665           # gravity (m/s^2)
_M = 0.0289644         # molar mass of dry air (kg/mol)
_R = 8.3144598         # universal gas constant (J/(mol*K))
_EXP = _G * _M / (_R * _L)   # ~5.25588


def altitude_to_pressure_hpa(altitude_m: float) -> float:
    """ISA pressure (hPa) at a geopotential altitude (m above sea level)."""
    return _P0_HPA * (1.0 - _L * altitude_m / _T0_K) ** _EXP


def pressure_to_altitude_m(pressure_hpa: float) -> float:
    """ISA altitude (m above sea level) for an absolute pressure (hPa). Inverse of above."""
    return (_T0_K / _L) * (1.0 - (pressure_hpa / _P0_HPA) ** (1.0 / _EXP))
