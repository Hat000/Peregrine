import pytest

from racer.atmosphere import altitude_to_pressure_hpa, pressure_to_altitude_m


def test_sea_level():
    assert altitude_to_pressure_hpa(0.0) == pytest.approx(1013.25)
    assert pressure_to_altitude_m(1013.25) == pytest.approx(0.0, abs=1e-6)


def test_known_value_1000m():
    # ISA pressure at 1000 m is ~898.7 hPa.
    assert altitude_to_pressure_hpa(1000.0) == pytest.approx(898.76, abs=0.5)


def test_round_trip():
    for h in (-50.0, 0.0, 1.8, 30.0, 500.0, 3000.0):
        assert pressure_to_altitude_m(altitude_to_pressure_hpa(h)) == pytest.approx(h, abs=1e-6)


def test_pressure_decreases_with_altitude():
    assert altitude_to_pressure_hpa(100.0) < altitude_to_pressure_hpa(0.0)
