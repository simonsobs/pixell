"""Public gal,cel path must match coordinates.transform ICRS."""
import unittest
import numpy as np
from pixell import coordinates, curvedsky, reproject, utils


class GalcelIcrsTests(unittest.TestCase):
    def test_rot2euler_gal_cel_matches_coordinates_icrs(self):
        ra, dec = np.deg2rad([4.442941, 4.312485])
        pos = np.array([[ra], [dec]])
        via_coordinates = coordinates.transform("equ", "gal", pos)
        euler = reproject.inv_euler(reproject.rot2euler("gal,cel"))[::-1]
        via_reproject = coordinates.transform_euler(euler, pos, pol=False)
        lon1, lat1 = via_coordinates[:, 0]
        lon2, lat2 = via_reproject[:, 0]
        separation = np.arccos(
            np.clip(
                np.sin(lat1) * np.sin(lat2)
                + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2),
                -1,
                1,
            )
        )
        self.assertLess(float(separation / utils.arcsec), 1e-3)

    def test_curvedsky_gal_equ_euler_matches_rot2euler(self):
        np.testing.assert_allclose(
            curvedsky.euler_angs[("gal", "equ")],
            reproject.rot2euler("gal,cel"),
            rtol=0,
            atol=1e-15,
        )
