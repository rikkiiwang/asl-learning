import numpy as np
from aslv2.geometry import slot_hands


def _box(cx, cy, s):
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


def test_slots_are_stable_when_box_areas_swap():
    head_cx = 100.0
    # left hand stays at x=60, right at x=140; their SIZES swap across frames.
    frames = [
        np.array([_box(60, 50, 20), _box(140, 50, 40)]),   # right bigger
        np.array([_box(60, 50, 40), _box(140, 50, 20)]),   # left bigger (area swap)
    ]
    slots, present = slot_hands(frames, head_cx)
    # slot 0 (left) center.x must stay ~60 in BOTH frames despite the area swap
    for f in range(2):
        lcx = (slots[f, 0, 0] + slots[f, 0, 2]) / 2
        assert abs(lcx - 60) < 1e-6
    assert present.tolist() == [[1, 1], [1, 1]]


def test_single_hand_goes_to_side_relative_to_head():
    head_cx = 100.0
    frames = [np.array([_box(40, 50, 20)])]            # one hand, left of head
    slots, present = slot_hands(frames, head_cx)
    assert present.tolist() == [[1, 0]]                # left slot filled
    assert np.isnan(slots[0, 1]).all()                 # right slot empty

    frames = [np.array([_box(160, 50, 20)])]           # one hand, right of head
    slots, present = slot_hands(frames, head_cx)
    assert present.tolist() == [[0, 1]]
