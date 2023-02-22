"""Configuration for various execution options and defaults."""

import numpy as np

SHAPEDNA_DEFAULTS: dict = {"Refine": 0, "Degree": 1}

tb = (
    np.array(
        [
            [8.0, 4.0, 2.0, 4.0, 4.0, 2.0, 1.0, 2.0],
            [4.0, 8.0, 4.0, 2.0, 2.0, 4.0, 2.0, 1.0],
            [2.0, 4.0, 8.0, 4.0, 1.0, 2.0, 4.0, 2.0],
            [4.0, 2.0, 4.0, 8.0, 2.0, 1.0, 2.0, 4.0],
            [4.0, 2.0, 1.0, 2.0, 8.0, 4.0, 2.0, 4.0],
            [2.0, 4.0, 2.0, 1.0, 4.0, 8.0, 4.0, 2.0],
            [1.0, 2.0, 4.0, 2.0, 2.0, 4.0, 8.0, 4.0],
            [2.0, 1.0, 2.0, 4.0, 4.0, 2.0, 4.0, 8.0],
        ]
    )
    / 216.0
)
x = 1.0 / 9.0
y = 1.0 / 18.0
z = 1.0 / 36.0
ta00 = np.array(
    [
        [x, -x, -y, y, y, -y, -z, z],
        [-x, x, y, -y, -y, y, z, -z],
        [-y, y, x, -x, -z, z, y, -y],
        [y, -y, -x, x, z, -z, -y, y],
        [y, -y, -z, z, x, -x, -y, y],
        [-y, y, z, -z, -x, x, y, -y],
        [-z, z, y, -y, -y, y, x, -x],
        [z, -z, -y, y, y, -y, -x, x],
    ]
)
ta11 = np.array(
    [
        [x, y, -y, -x, y, z, -z, -y],
        [y, x, -x, -y, z, y, -y, -z],
        [-y, -x, x, y, -z, -y, y, z],
        [-x, -y, y, x, -y, -z, z, y],
        [y, z, -z, -y, x, y, -y, -x],
        [z, y, -y, -z, y, x, -x, -y],
        [-z, -y, y, z, -y, -x, x, y],
        [-y, -z, z, y, -x, -y, y, x],
    ]
)
ta22 = np.array(
    [
        [x, y, z, y, -x, -y, -z, -y],
        [y, x, y, z, -y, -x, -y, -z],
        [z, y, x, y, -z, -y, -x, -y],
        [y, z, y, x, -y, -z, -y, -x],
        [-x, -y, -z, -y, x, y, z, y],
        [-y, -x, -y, -z, y, x, y, z],
        [-z, -y, -x, -y, z, y, x, y],
        [-y, -z, -y, -x, y, z, y, x],
    ]
)
