"""Tests for landmark inference + frame mapping."""
import numpy as np
import pytest
import torch


def test_landmark_hand_output_shape_and_range():
    """landmark_hand returns (21,2) ndarray with values in [0,1]."""
    from aslv2.landmark.model import Landmark
    from aslv2.landmark.infer import landmark_hand

    model = Landmark().eval()
    crop_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}

    kp = landmark_hand(model, crop_rgb, norm)

    assert kp.shape == (21, 2), f"Expected (21,2), got {kp.shape}"
    assert float(kp.min()) >= 0.0, "kp values must be >= 0"
    assert float(kp.max()) <= 1.0, "kp values must be <= 1"


def test_map_to_frame_centre():
    """[0.5,0.5] with box [10,20,50,60] maps to [30,40]."""
    from aslv2.landmark.infer import map_to_frame

    kp01 = np.array([[0.5, 0.5]], dtype=np.float32)
    box_xyxy = np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32)
    result = map_to_frame(kp01, box_xyxy)

    assert result.shape == (1, 2), f"Expected (1,2), got {result.shape}"
    np.testing.assert_allclose(result[0], [30.0, 40.0], atol=0.5,
                               err_msg="Centre keypoint should map to box centre")


def test_map_to_frame_corners():
    """[0,0] → box top-left; [1,1] → box bottom-right."""
    from aslv2.landmark.infer import map_to_frame

    box_xyxy = np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32)
    kp01 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    result = map_to_frame(kp01, box_xyxy)

    np.testing.assert_allclose(result[0], [10.0, 20.0], atol=1e-5)
    np.testing.assert_allclose(result[1], [50.0, 60.0], atol=1e-5)
