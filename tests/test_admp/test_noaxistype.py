import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from dmff.admp.spatial import (build_quasi_internal,
                          generate_construct_local_frames, pbc_shift,
                          v_pbc_shift)


class TestSpatial:

    @pytest.mark.parametrize(
        "axis_types, axis_indices, positions, box, expected_local_frames",
        [
            (
                np.array([5]),
                np.array(
                    [
                        [-1, -1, -1],
                    ]
                ),
                jnp.array(
                    [
                        [0.992, 0.068, -0.073],
                    ]
                ),
                jnp.array([[50.000, 0.0, 0.0], [0.0, 50.000, 0.0], [0.0, 0.0, 50.000]]),
                np.array(
                    [
                        [
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
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

