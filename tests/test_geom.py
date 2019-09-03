import unittest

from pixell import enmap, wcsutils, utils
DEG = utils.degree
import numpy as np

def is_centered(pixel_index, rtol=1e-6):
    """Returns element-by-element True if pixel_index is sufficiently
    close to integer-valued."""
    frac = (np.asarray(pixel_index) + 0.5) % 1.0 - 0.5
    return np.isclose(frac, 0., rtol=rtol)

class Patch:
    ra_range = None  #! Right Ascension; left side, right side.
    dec_range = None #! Declination; bottom, top.
    @classmethod
    def centered_at(cls, ra0, dec0, width, height):
        self = cls()
        self.ra_range = (ra0+width/2, ra0-width/2)
        self.dec_range = (dec0-height/2, dec0+height/2)
        return self

    def pos(self):
        return np.array([[self.dec_range[0], self.ra_range[0]],
                         [self.dec_range[1], self.ra_range[1]]]) * DEG
    def extent(self, delta_ra = 0., delta_dec = 0.):
        return [self.ra_range[0]  + delta_ra /2, self.ra_range[1]  - delta_ra /2,
                self.dec_range[0] - delta_dec/2, self.dec_range[1] + delta_dec/2]
    def center(self):
        return 0.5 * np.array([(self.dec_range[0] + self.dec_range[1]),
                               (self.ra_range[0] + self.ra_range[1])])


class GeometryTests(unittest.TestCase):

    def test_reference(self):
        """Test that WCS are properly adjusted, on request, to put a reference
        pixel at integer pixel number.

        """
        DELT = 0.1
        # Note we're adding a half-pixel margin to stay away from rounding cut.
        patch = Patch.centered_at(-52., -38., 12. + DELT, 12.0 + DELT)
        # Test a few reference RAs.
        for ra1 in [0., 0.001, 90.999, 91.101, 120., 179.9, 180.0, 180.1, 270.]:
            shape, wcs = enmap.geometry(pos=patch.pos(),
                                        res=DELT*DEG,
                                        proj='cea',
                                        ref=(0, ra1*DEG))
            ref = np.array([[ra1,0]])
            ref_pix = wcs.wcs_world2pix(ref, 0)
            assert(np.all(is_centered(ref_pix)))

    def test_zenithal(self):
        DELT = 0.05
        patch0 = Patch.centered_at(308., -38., 1.+DELT, 1.+DELT)
        patch1 = Patch.centered_at(309., -39., 1.+DELT, 1.+DELT)
        ref = patch0.center()   # (dec,ra) in degrees.
        for proj in ['tan', 'zea', 'air']:
            print('Checking reference tracking of "%s"...' % proj)
            shape0, wcs0 = enmap.geometry(pos=patch0.pos(),
                                          res=DELT*utils.degree,
                                          proj=proj,
                                          ref=ref*DEG)
            shape1, wcs1 = enmap.geometry(pos=patch1.pos(),
                                          res=DELT*utils.degree,
                                          proj=proj,
                                          ref=ref*DEG)
            # Note world2pix wants [(ra,dec)], in degrees...
            pix0 = wcs0.wcs_world2pix(ref[::-1][None],0)
            pix1 = wcs1.wcs_world2pix(ref[::-1][None],0)
            print(shape0,wcs0,pix0)
            assert(np.all(is_centered(pix0)))
            assert(np.all(is_centered(pix1)))

if __name__ == '__main__':
    unittest.main()
