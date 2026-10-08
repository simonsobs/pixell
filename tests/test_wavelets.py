import unittest

import numpy as np

from pixell import curvedsky, enmap, uharm, utils, wavelets

LPEAKS = [0, 50, 100, 200, 300]


def curved_wt():
    shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin)
    uht = uharm.UHT(shape, wcs, mode="curved")
    return wavelets.WaveletTransform(uht, basis=wavelets.CosineNeedlet(lpeaks=LPEAKS))


def flat_wt():
    shape, wcs = enmap.geometry(np.array([[-5, -5], [5, 5]]) * utils.degree, res=2 * utils.arcmin)
    basis = wavelets.CosineNeedlet(lpeaks=[0, 300, 600, 1200, 2400, 4800])
    return wavelets.WaveletTransform(uharm.UHT(shape, wcs, mode="flat"), basis=basis)


def rand_map(shape, wcs):
    # One component: for maps with pre-dimensions, map2wave uses spin [0, 2] alms on the
    # curved sky and scalar Fourier transforms on the flat sky, while uht uses spin 0
    return enmap.rand_gauss(shape[-2:], wcs, dtype=np.float64)


def assert_wave_close(wa, wb, rtol=1e-10):
    for a, b in zip(wa.maps, wb.maps):
        np.testing.assert_allclose(a, b, rtol=0, atol=rtol * np.abs(b).max())


class WaveletTests(unittest.TestCase):
    def test_harm2wave_matches_map2wave(self):
        for wt in (curved_wt(), flat_wt()):
            imap = rand_map(*wt.geometry)
            if wt.uht.mode == "curved":
                # Band-limited to the basis, so that the alms of uht (to uht.lmax) and of
                # map2wave (to basis.lmax) agree below basis.lmax
                imap = wt.uht.harm2map(curvedsky.transfer_alm(
                    curvedsky.alm_info(lmax=wt.basis.lmax),
                    curvedsky.rand_alm(np.ones(wt.basis.lmax + 1), seed=3),
                    wt.uht.ainfo,
                ))
            assert_wave_close(wt.harm2wave(wt.uht.map2harm(imap)), wt.map2wave(imap))

    def test_harm2wave_filter(self):
        wt = curved_wt()
        imap = rand_map(*wt.geometry)
        alm = curvedsky.map2alm(imap, lmax=wt.basis.lmax)
        fl = np.exp(-np.arange(wt.basis.lmax + 1) / 200.0)
        assert_wave_close(wt.harm2wave(alm, fl=fl), wt.map2wave(imap, fl=fl))

    def test_harm2wave_any_lmax(self):
        wt = curved_wt()
        lmax = wt.basis.lmax
        alm = curvedsky.rand_alm(np.ones(lmax + 101), lmax=lmax + 100, seed=1)
        cut = curvedsky.transfer_alm(
            curvedsky.alm_info(lmax=lmax + 100), alm, curvedsky.alm_info(lmax=lmax)
        )
        # Multipoles above basis.lmax are ignored
        assert_wave_close(wt.harm2wave(alm), wt.harm2wave(cut))

    def test_scales_fill_value(self):
        wt = curved_wt()
        alm = curvedsky.rand_alm(np.ones(wt.basis.lmax + 1), seed=2)
        wave = wt.harm2wave(alm, scales=[0], fill_value=7.0)
        self.assertTrue(np.all(wave.maps[1] == 7.0))
        self.assertFalse(np.all(wave.maps[0] == 7.0))
        wt = flat_wt()
        wave = wt.map2wave(rand_map(*wt.geometry), scales=[0], fill_value=7.0)
        self.assertTrue(np.all(wave.maps[1] == 7.0))
        self.assertFalse(np.all(wave.maps[0] == 7.0))

    def test_geometry(self):
        wt = curved_wt()
        shape, wcs = wt.geometry
        self.assertEqual(shape, wt.uht.shape)
        self.assertIs(wcs, wt.uht.wcs)

    def test_wave2harm_matches_wave2map(self):
        for wt in (curved_wt(), flat_wt()):
            wave = wt.map2wave(rand_map(*wt.geometry))
            omap = wt.wave2map(wave)
            harm = wt.wave2harm(wave)
            if wt.uht.mode == "curved":
                # alms to basis.lmax, padded with zeros to the lmax of uht
                self.assertEqual(curvedsky.nalm2lmax(harm.shape[-1]), wt.basis.lmax)
                harm = curvedsky.transfer_alm(
                    curvedsky.alm_info(lmax=wt.basis.lmax), harm, wt.uht.ainfo
                )
            np.testing.assert_allclose(
                wt.uht.harm2map(harm), omap, rtol=0, atol=1e-10 * np.abs(omap).max()
            )

    def test_flat_roundtrip(self):
        # The default basis covers all multipoles of the map, so the transform is invertible
        shape, wcs = flat_wt().geometry
        wt = wavelets.WaveletTransform(uharm.UHT(shape, wcs, mode="flat"))
        imap = rand_map(shape, wcs)
        omap = wt.wave2map(wt.map2wave(imap))
        self.assertTrue(np.all(np.isfinite(omap)))
        np.testing.assert_allclose(omap, imap, rtol=0, atol=1e-10 * np.abs(imap).max())

    def test_resample_fft_accumulate(self):
        # Summing into fomap with op=np.add must not re-shift what is already there
        imap = rand_map(*flat_wt().geometry)
        fmap = enmap.fft(imap, normalize=False)
        oshape = (imap.shape[-2] // 2, imap.shape[-1] // 2)
        once = enmap.resample_fft(fmap, oshape, corner=True)
        twice = enmap.resample_fft(fmap, oshape, corner=True)
        enmap.resample_fft(fmap, oshape, fomap=twice, corner=True, op=np.add)
        self.assertTrue(np.any(once != 0))
        np.testing.assert_allclose(twice, 2 * once, rtol=0, atol=1e-12 * np.abs(once).max())

    def test_flat_filter_not_implemented(self):
        wt = flat_wt()
        with self.assertRaises(NotImplementedError):
            wt.harm2wave(wt.uht.map2harm(rand_map(*wt.geometry)), fl=np.ones(10))


if __name__ == "__main__":
    unittest.main()
