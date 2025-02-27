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
        self.ra_range = np.array((ra0+width/2, ra0-width/2))
        self.dec_range = np.array((dec0-height/2, dec0+height/2))
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

    def test_zenithal_lonpole(self):
        """For zenithal projs, test that lonpole is set to 180, even
        if the reference point is the north pole.

        """
        DELT = 0.05
        for geometry_gen in [enmap.geometry, enmap.geometry2]:
            for ep in [DELT * 2, 0]:
                patch = Patch.centered_at(0., 90.-ep, 0., 0.)
                for proj in ['tan', 'zea', 'arc', 'sin']:
                    shape, wcs = geometry_gen(pos=patch.center() * DEG,
                                              shape=(101, 101),
                                              res=DELT*utils.degree,
                                              proj=proj)
                    dec, ra = enmap.posmap(shape, wcs)
                    # Bottom row of the map only have RA near 0 (not 180).
                    assert np.all(abs(ra[0]) < 90*DEG)

    def test_full_sky(self):
        """Test that fullsky_geometry returns sensible objects.

        Or at least the objects that we considered sensible when we
        wrote this test.

        """
        shape, w = enmap.fullsky_geometry(res=0.01*DEG, proj='car')
        ny, nx = shape
        for delta, expect_nans in [(0., False), (.001, True)]:
            for ix in [-0.5-delta,nx-0.5+delta]:
                for iy in [0-delta, ny-1+delta]:
                    c = w.wcs_pix2world([(ix,iy)], 0)
                    #print(ix,iy,c)
                    assert np.any(np.isnan(c)) == expect_nans


    def test_area(self):
        """Test that map area is computed accurately."""
        test_patches = []
        # Small CAR patch
        DELT = 0.01
        patch = Patch.centered_at(-52., -38., 12. + DELT, 12.0 + DELT)
        shape, w = enmap.geometry(pos=patch.pos(),
                                  res=DELT*DEG,
                                  proj='car',
                                  ref=(0, 0))
        exact_area = (np.dot(patch.ra_range*DEG, [-1,1]) *
                      np.dot(np.sin(patch.dec_range*DEG), [1,-1]))

        test_patches.append((shape, w, exact_area))
        # Full sky CAR patch
        shape, w = enmap.fullsky_geometry(res=0.01*DEG, proj='car')
        exact_area = 4*np.pi
        test_patches.append((shape, w, exact_area))
        # Small ZEA patch at pole
        shape, w = enmap.geometry(pos=[90*DEG,0], res=DELT*DEG, proj='zea', shape=[100,100])
        exact_area = 1*DEG**2
        test_patches.append((shape, w, exact_area))

        for shape, w, exact_area in test_patches:
            ratio = enmap.area(shape, w)/exact_area
            print(ratio)
            assert(abs(ratio-1) < 1e-6)

    def test_geom_args(self):
        """Test that the different combinations of passing and not passing arguments
        to geometry() work."""
        box = np.array([[40,100],[45,95]])*utils.degree
        res = 1*utils.degree
        def close(a,b): return np.all(np.isclose(a,b,atol=1e-6*utils.degree))
        # Plain
        shape, wcs = enmap.geometry(box, res=res, proj="car")
        assert(close(shape, [5,5]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.00000000,100.00000000]))
        shape, wcs = enmap.geometry(box, res=res, proj="cea")
        assert(close(shape, [7,5]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.17551345,100.00000000]))
        shape, wcs = enmap.geometry(box, res=res, proj="zea")
        assert(close(shape, [5,4]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.49221947, 98.81479953]))
        shape, wcs = enmap.geometry(box, res=res, proj="tan")
        assert(close(shape, [5,4]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.49336943, 98.81407198]))
        shape, wcs = enmap.geometry(box, res=res, proj="air")
        assert(close(shape, [2,10]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.42085960, 99.75799905]))
        shape, wcs = enmap.geometry(box, res=res, proj="plain")
        assert(close(shape, [5,5]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.50000000, 99.50000000]))
        # Swap box order
        shape, wcs = enmap.geometry(box[::-1], res=res, proj="car")
        assert(close(shape, [5,5]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [45.00000000, 95.00000000]))
        shape, wcs = enmap.geometry(box[::-1], res=res, proj="cea")
        assert(close(shape, [7,5]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [44.59207383, 95.00000000]))
        shape, wcs = enmap.geometry(box[::-1], res=res, proj="zea")
        assert(close(shape, [5,4]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [44.49175753, 96.09829319]))
        shape, wcs = enmap.geometry(box[::-1], res=res, proj="tan")
        assert(close(shape, [5,4]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [44.49062589, 96.09912004]))
        shape, wcs = enmap.geometry(box[::-1], res=res, proj="air")
        assert(close(shape, [2,10]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [44.88532797, 95.12307674]))
        shape, wcs = enmap.geometry(box[::-1], res=res, proj="plain")
        assert(close(shape, [5,5]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [44.50000000, 95.50000000]))
        # Specify center and shape instead of box. FIXME: These should be changed to make them right-handed.
        pos   = np.array([50,120])*utils.degree
        shape = (5,5)
        oshape, wcs = enmap.geometry(pos, res=res, shape=shape, proj="car")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [48.00000000,122.00000000]))
        oshape, wcs = enmap.geometry(pos, res=res, shape=shape, proj="cea")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [48.58804987,122.00000000]))
        oshape, wcs = enmap.geometry(pos, res=res, shape=shape, proj="zea")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [47.96024416,122.98709530]))
        oshape, wcs = enmap.geometry(pos, res=res, shape=shape, proj="tan")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [47.96213990,122.98447863]))
        oshape, wcs = enmap.geometry(pos, res=res, shape=shape, proj="plain")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [48.00000000,118.00000000]))
        # Same but with explicit 2d res
        oshape, wcs = enmap.geometry(pos, res=[res,-res], shape=shape, proj="car")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [48.00000000,122.00000000]))
        oshape, wcs = enmap.geometry(pos, res=[res,-res], shape=shape, proj="cea")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [48.58804987,122.00000000]))
        oshape, wcs = enmap.geometry(pos, res=[res,-res], shape=shape, proj="zea")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [47.96024416,122.98709530]))
        oshape, wcs = enmap.geometry(pos, res=[res,-res], shape=shape, proj="tan")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [47.96213990,122.98447863]))
        oshape, wcs = enmap.geometry(pos, res=[res,-res], shape=shape, proj="plain")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [48.00000000,122.00000000]))
        # Box and shape
        oshape, wcs = enmap.geometry(box, shape=shape, proj="car")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.00000000,100.00000000]))
        oshape, wcs = enmap.geometry(box, shape=shape, proj="cea")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.03023148,100.00000000]))
        oshape, wcs = enmap.geometry(box, shape=shape, proj="zea")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.48400209, 99.43687135]))
        oshape, wcs = enmap.geometry(box, shape=shape, proj="tan")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.48319549, 99.43761826]))
        oshape, wcs = enmap.geometry(box, shape=shape, proj="air")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.04076497,100.02606306]))
        oshape, wcs = enmap.geometry(box, shape=shape, proj="plain")
        assert(close(oshape, shape))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [40.50000000, 99.50000000]))

    def test_thumb_geom_args(self):
        """Test that the different combinations of passing and not passing arguments
        to thumbnail_geometry() work."""
        r   = 5*utils.degree
        res = 1*utils.degree
        def close(a,b): return np.all(np.isclose(a,b,atol=1e-6*utils.degree))
        shape, wcs = enmap.thumbnail_geometry(r=r, res=res, proj="car")
        assert(close(shape, [11,11]))
        #assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-5.00000000, -5.00000000]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-5.00000000,  5.00000000]))
        shape, wcs = enmap.thumbnail_geometry(r=r, res=res, proj="cea")
        assert(close(shape, [11,11]))
        #assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-5.00636804, -5.00000000  ]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-5.00636804,  5.00000000  ]))
        shape, wcs = enmap.thumbnail_geometry(r=r, res=res, proj="zea")
        assert(close(shape, [11,11]))
        #assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-4.99680325, -5.01591450]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-4.99680325,  5.01591450]))
        shape, wcs = enmap.thumbnail_geometry(r=r, res=res, proj="air")
        assert(close(shape, [11,11]))
        #assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-4.99205428, -5.01111089]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-4.99205428,  5.01111089]))
        shape, wcs = enmap.thumbnail_geometry(r=r, res=res, proj="tan")
        assert(close(shape, [11,11]))
        #assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-4.96857729, -4.98736529]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-4.96857729,  4.98736529]))
        shape, wcs = enmap.thumbnail_geometry(r=r, res=res, proj="plain")
        assert(close(shape, [11,11]))
        #assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-5.00000000, -5.00000000]))
        assert(close(enmap.pix2sky(shape, wcs, [0,0])/utils.degree, [-5.00000000, -5.00000000]))

if __name__ == '__main__':
    unittest.main()
