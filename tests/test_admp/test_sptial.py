import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from dmff.admp.spatial import (build_quasi_internal,
                          generate_construct_local_frames, pbc_shift,
                          v_pbc_shift)


class TestSpatial:
    @pytest.mark.parametrize(
        "r1, r2, dr, norm_dr, expected",
        [
            (
                jnp.array([[0, 0, 0]]),
                jnp.array([[1, 0, 0]]),
                jnp.array([[1, 0, 0]]),
                jnp.array([1.0,]),
                jnp.array([[[0.0, 1.0, 0.0], [0, 0, 1], [1, 0, 0]]]),
            ),
            (
                jnp.array([[0, 0, 0]]),
                jnp.array([[1, 1, 0]]),
                jnp.array([[1, 1, 0]]),
                jnp.array([1.414213,]),
                jnp.array(
                    [
                        [
                            [0.70710534, -0.70710814, 0.0],
                            [0.0, 0.0, -1.0000004],
                            [0.70710707, 0.70710707, 0.0],
                        ]
                    ]
                ),
            ),
        ],
    )
    def test_build_quasi_internal(self, r1, r2, dr, norm_dr, expected):

        local_frames = build_quasi_internal(r1, r2, dr, norm_dr)
        npt.assert_allclose(local_frames, expected, rtol=1e-5)

    @pytest.mark.parametrize(
        "drvecs, box, box_inv, expected",
        [
            (
                jnp.array([[0, 0, 0]]),
                jnp.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
                jnp.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]]),
                jnp.array([[0, 0, 0]]),
            ),
            (
                jnp.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
                jnp.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
                jnp.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]]),
                jnp.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            ),
            (
                jnp.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
                jnp.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
                jnp.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]]),
                jnp.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]]),
            ),
        ],
    )
    def test_pbc_shift(self, drvecs, box, box_inv, expected):
        npt.assert_allclose(pbc_shift(drvecs, box, box_inv), expected)
        npt.assert_allclose(v_pbc_shift(drvecs, box, box_inv), expected)

    @pytest.mark.parametrize(
        "axis_types, axis_indices, positions, box, expected_local_frames",
        [
            (
                np.array([1, 0, 0, 1, 0, 0]),
                np.array(
                    [
                        [1, 2, -1],
                        [0, 2, -1],
                        [0, 1, -1],
                        [4, 5, -1],
                        [3, 5, -1],
                        [3, 4, -1],
                    ]
                ),
                jnp.array(
                    [
                        [1.562, 24.46, 21.149],
                        [2.439, 24.438, 21.532],
                        [0.983, 24.178, 21.854],
                        [4.788, 24.609, 0.994],
                        [4.026, 25.138, 1.231],
                        [4.462, 23.712, 0.97],
                    ]
                ),
                jnp.array([[31.289, 0.0, 0.0], [0.0, 31.289, 0.0], [0.0, 0.0, 31.289]]),
                np.array(
                    [
                        [
                            [-0.96165454, -0.17201543, 0.21361469],
                            [0.10460715, -0.95003253, -0.29410106],
                            [0.2535308, -0.26047802, 0.9315972],
                        ],
                        [
                            [-0.38687626, -0.3113036, 0.8679958],
                            [-0.10460713, 0.9500325, 0.2941011],
                            [-0.91617906, 0.02298216, -0.4001096],
                        ],
                        [
                            [0.7882788, -0.10109846, 0.606956],
                            [0.10460714, -0.95003265, -0.29410112],
                            [0.60636103, 0.29532564, -0.7383151],
                        ],
                        [
                            [0.2869897, -0.94232714, -0.17221032],
                            [0.22607784, -0.10806667, 0.96809626],
                            [-0.93087363, -0.3167666, 0.18202528],
                        ],
                        [
                            [-0.5616504, -0.8264594, 0.03890521],
                            [-0.22607785, 0.10806668, -0.9680963],
                            [0.79588807, -0.5525272, -0.24753988],
                        ],
                        [
                            [-0.9122986, 0.32489008, 0.24931434],
                            [0.22607782, -0.10806666, 0.9680962],
                            [0.3414673, 0.9395574, 0.02513866],
                        ],
                    ]
                ),
            )
        ],
    )
    def test_generate_construct_local_frames(
        self, axis_types, axis_indices, positions, box, expected_local_frames
    ):
        construct_local_frame_fn = generate_construct_local_frames(
            axis_types, axis_indices
        )
        assert construct_local_frame_fn
        npt.assert_allclose(
            construct_local_frame_fn(positions, box), expected_local_frames, rtol=1e-5
        )

