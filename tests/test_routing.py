import pytest

from src.pipeline.routing import Thresholds, route


def test_route_pass():
    assert route(0.05, Thresholds(0.30, 0.80)) == "pass"


def test_route_block():
    assert route(0.95, Thresholds(0.30, 0.80)) == "block"


def test_route_escalate():
    assert route(0.50, Thresholds(0.30, 0.80)) == "escalate"


def test_route_boundary_unsafe_is_block():
    assert route(0.80, Thresholds(0.30, 0.80)) == "block"


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        Thresholds(0.8, 0.3)
