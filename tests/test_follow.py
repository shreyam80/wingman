"""Automatic checks for the follow pieces. Run:  .venv/bin/python -m pytest -q
These test behaviour that matters for safety, without any video or model."""
import json
from pathlib import Path

import numpy as np
import pytest

from wingman.follow.config import load_config
from wingman.follow.controller import VerticalController
from wingman.follow.detector import TargetLock
from wingman.follow.drone import SimDrone
from wingman.follow.geometry import WallMap

CFG, _ = load_config()
W, H = 600, 960


# --- simulated drone -----------------------------------------------------------
def test_drone_never_exceeds_max_speed():
    d = SimDrone(CFG, 0.0, 2.0)
    d.send_up_speed(100.0)  # absurd command
    speeds = []
    for _ in range(200):
        d.step(0.04)
        speeds.append(abs(d.vy))
    assert max(speeds) <= CFG.controller.max_speed_mps + 1e-9


def test_drone_ignores_command_until_latency_passed():
    d = SimDrone(CFG, 0.0, 2.0)
    d.send_up_speed(0.5)
    d.step(CFG.loop.latency_s * 0.5)
    assert d.vy == 0.0
    d.step(CFG.loop.latency_s)
    assert d.vy > 0.0


def test_drone_stops_at_height_limit():
    d = SimDrone(CFG, 0.0, CFG.sim.max_view_center_m - 0.01)
    d.send_up_speed(0.5)
    for _ in range(100):
        d.step(0.04)
    assert d.y == CFG.sim.max_view_center_m and d.vy == 0.0


# --- controller ----------------------------------------------------------------
def test_no_target_means_hold():
    c = VerticalController(CFG, view_height_m=4.0)
    assert c.update(None, 0.1) == 0.0


def test_small_offset_inside_deadband_means_hold():
    c = VerticalController(CFG, view_height_m=4.0)
    assert c.update(CFG.controller.target_point_frac - CFG.controller.deadband_frac / 2, 0.1) == 0.0


def test_climber_above_target_means_go_up_and_below_means_down():
    up = VerticalController(CFG, 4.0)
    down = VerticalController(CFG, 4.0)
    for _ in range(20):
        u, dn = up.update(0.2, 0.1), down.update(0.8, 0.1)  # 0 = top of picture
    assert u > 0 and dn < 0


def test_controller_respects_speed_and_acceleration_limits():
    c = VerticalController(CFG, 4.0)
    outs = [c.update(0.0, 0.1) for _ in range(50)]
    assert max(abs(o) for o in outs) <= CFG.controller.max_speed_mps + 1e-9
    assert max(abs(b - a) for a, b in zip([0.0] + outs, outs)) <= CFG.controller.max_accel_mps2 * 0.1 + 1e-9


# --- target lock ---------------------------------------------------------------
def box(cx, cy, h=200):
    return (cx - 40, cy - h / 2, cx + 40, cy + h / 2)


def test_locks_person_nearest_centre_not_biggest():
    lock = TargetLock(CFG, W, H)
    b, ev = lock.update(0.0, [(1, box(300, 480)), (2, box(80, 850, h=600))])
    assert ev == "lock" and lock.track_id == 1


def test_lost_then_same_id_returns():
    lock = TargetLock(CFG, W, H)
    lock.update(0.0, [(1, box(300, 480))])
    assert lock.update(0.1, []) == (None, "lost")
    b, ev = lock.update(0.2, [(1, box(300, 470))])
    assert ev == "relock" and lock.state == "LOCKED"


def test_never_switches_to_far_away_person():
    lock = TargetLock(CFG, W, H)
    lock.update(0.0, [(1, box(300, 480))])
    lock.update(0.1, [])
    b, ev = lock.update(0.2, [(7, box(80, 850))])  # bystander elsewhere in the picture
    assert b is None and lock.state == "LOST"


def test_reacquires_new_id_close_to_last_position_within_window():
    lock = TargetLock(CFG, W, H)
    lock.update(0.0, [(1, box(300, 480))])
    lock.update(0.1, [])
    b, ev = lock.update(0.5, [(9, box(310, 490))])
    assert ev == "relock" and lock.track_id == 9


def test_asks_pilot_after_window_and_holds():
    lock = TargetLock(CFG, W, H)
    lock.update(0.0, [(1, box(300, 480))])
    lock.update(0.1, [])
    b, ev = lock.update(0.1 + CFG.lock.reacquire_window_s + 0.5, [(9, box(310, 490))])
    assert b is None and ev == "needs_pilot"


# --- wall geometry (needs the dataset) ------------------------------------------
DATASET = Path("~/Downloads/wingman dataset").expanduser()


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not on this computer")
def test_wallmap_puts_reference_bolts_at_their_wall_positions():
    f = DATASET / "p1/orange_bolt_holes.json"
    wm = WallMap(f)
    b = json.load(open(f))
    got = wm.to_wall(b["img_coords"])
    want = np.array(b["world_coords"])[:, :2] / 100
    # rows are matched left-to-right, so compare as sorted sets per row
    for y in np.unique(want[:, 1]):
        g = got[np.isclose(got[:, 1], y, atol=0.05)]
        assert len(g) == (want[:, 1] == y).sum()
