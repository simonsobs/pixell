import unittest

import numpy as np

from pixell import curvedsky, enmap, uharm, utils, wavelets

LPEAKS = [0, 50, 100, 200, 300]


def curved_wt():
    shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin)
    uht = uharm.UHT(shape, wcs, mode="curved")
    return wavelets.WaveletTransform(uht, basis=wavelets.CosineNeedlet(lpeaks=LPEAKS))


def rand_map(shape, wcs):
    return enmap.rand_gauss((3,) + shape[-2:], wcs, dtype=np.float64)


def assert_wave_close(wa, wb, rtol=1e-10):
    for a, b in zip(wa.maps, wb.maps):
        np.testing.assert_allclose(a, b, rtol=0, atol=rtol * np.abs(b).max())


class WaveletTests(unittest.TestCase):
    def test_alm2wave_matches_map2wave(self):
        wt = curved_wt()
        imap = rand_map(*wt.geometry)
        alm = curvedsky.map2alm(imap, lmax=wt.basis.lmax)
        fl = np.exp(-np.arange(wt.basis.lmax + 1) / 200.0)
        assert_wave_close(wt.alm2wave(alm), wt.map2wave(imap))
        assert_wave_close(wt.alm2wave(alm, fl=fl), wt.map2wave(imap, fl=fl))

    def test_alm2wave_any_lmax(self):
        wt = curved_wt()
        lmax = wt.basis.lmax
        alm = curvedsky.rand_alm(np.ones(lmax + 101), lmax=lmax + 100, seed=1)
        cut = curvedsky.transfer_alm(
            curvedsky.alm_info(lmax=lmax + 100), alm, curvedsky.alm_info(lmax=lmax)
        )
        # Multipoles above basis.lmax are ignored
        assert_wave_close(wt.alm2wave(alm), wt.alm2wave(cut))

    def test_alm2wave_scales_fill_value(self):
        wt = curved_wt()
        alm = curvedsky.rand_alm(np.ones(wt.basis.lmax + 1), seed=2)
        wave = wt.alm2wave(alm, scales=[0], fill_value=7.0)
        self.assertTrue(np.all(wave.maps[1] == 7.0))
        self.assertFalse(np.all(wave.maps[0] == 7.0))

    def test_geometry(self):
        wt = curved_wt()
        shape, wcs = wt.geometry
        self.assertEqual(shape, wt.uht.shape)
        self.assertIs(wcs, wt.uht.wcs)

    def test_wave2alm_matches_wave2map(self):
        wt = curved_wt()
        wave = wt.map2wave(rand_map(*wt.geometry))
        oalm = wt.wave2alm(wave)
        self.assertEqual(curvedsky.nalm2lmax(oalm.shape[-1]), wt.basis.lmax)
        omap = curvedsky.alm2map(oalm, enmap.zeros(wave.pre + wt.shape[-2:], wt.wcs))
        np.testing.assert_allclose(omap, wt.wave2map(wave), rtol=0, atol=1e-12)

    def test_flat_fill_value(self):
        shape, wcs = enmap.geometry(np.array([[-5, -5], [5, 5]]) * utils.degree, res=2 * utils.arcmin)
        wt = wavelets.WaveletTransform(uharm.UHT(shape, wcs, mode="flat"))
        wave = wt.map2wave(enmap.rand_gauss(shape, wcs), scales=[0], fill_value=7.0)
        self.assertTrue(np.all(wave.maps[1] == 7.0))
        with self.assertRaises(NotImplementedError):
            wt.alm2wave(np.zeros(10, dtype=np.complex128))
        with self.assertRaises(NotImplementedError):
            wt.wave2alm(wave)


if __name__ == "__main__":
    unittest.main()
