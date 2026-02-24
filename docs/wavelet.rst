Wavelet analysis
================

Wavelets decompose a map into components that are simultaneously localized in both
real space (angular position) and harmonic space (multipole / angular scale). This
makes them powerful for tasks like multi-scale noise modeling, component separation,
and scale-dependent filtering.

Pixell provides the :py:class:`pixell.wavelets.WaveletTransform` class, which
implements a harmonic-space wavelet decomposition that works on
:py:class:`pixell.enmap.ndmap` objects in both flat-sky (FFT) and curved-sky (SHT)
modes. The transform is built on top of the :py:class:`pixell.uharm.UHT` unified
harmonic transform (see :doc:`harmonic <./harmonic>`).

Wavelet bases
--------------

Several basis types are available. All decompose the multipole range
:math:`[\ell_\text{min}, \ell_\text{max}]` into ``nlevel`` bands:

* **ButterTrim** (default): Butterworth band-pass filters with trimmed tails.
  Good spatial and harmonic localization; recommended for most use cases.
* **Butterworth**: Untrimmed Butterworth filters. Better spatial localization but
  tails extend to arbitrarily high :math:`\ell`.
* **DigitalButterTrim**: Digitized (orthogonal) version of ButterTrim.
* **CosineNeedlet**: Cosine-shaped needlets (Coulton et al. 2023), peaking at
  specified multipoles.
* **AdriSD**: Scale-discrete wavelet basis from the optweight library.

Basic usage
-----------

.. code-block:: python

    from pixell import enmap, uharm, wavelets, utils
    import numpy as np

    # Build a test map
    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=0.5 * utils.arcmin,
    )
    m = enmap.rand_gauss(shape, wcs)

    # 1. Create a UHT (Unified Harmonic Transform) for this geometry
    uht = uharm.UHT(shape, wcs)   # auto-selects flat or curved sky

    # 2. Create a wavelet transform with default ButterTrim basis
    wt = wavelets.WaveletTransform(uht)
    print(f"Number of wavelet scales: {wt.nlevel}")
    print(f"Scale mid-points (ell):   {wt.lmids}")

    # 3. Forward transform: map → wavelet coefficients (a multimap)
    wmap = wt.map2wave(m)
    print(f"Number of wavelet maps: {len(wmap.maps)}")
    for i, w in enumerate(wmap.maps):
        print(f"  scale {i}: shape={w.shape}, lmid={wt.lmids[i]:.0f}")

    # 4. Inverse transform: wavelet coefficients → map
    m_rec = wt.wave2map(wmap)
    print(f"Max roundtrip error: {np.max(np.abs(m_rec - m)):.2e}")

The wavelet coefficients are returned as a :py:mod:`pixell.multimap` — a
list-like object of enmaps where each map has the same leading dimensions but
different spatial resolution (lower-:math:`\ell` bands have larger pixels):

.. code-block:: python

    # Access individual wavelet scale maps
    w0 = wmap.maps[0]    # lowest-ell (large-scale) band
    wN = wmap.maps[-1]   # highest-ell (small-scale) band

    # Mathematical operations work on all scales simultaneously
    wmap_scaled = wmap * 2.0           # multiply all coefficients by 2
    wmap_noise  = wmap + wmap * 0.1   # add 10% noise to all scales

    #TODO: add figure -- run code:
    # from pixell import enplot
    # import matplotlib.pyplot as plt
    # n = wt.nlevel
    # fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    # for i, (w, ax) in enumerate(zip(wmap.maps, axes)):
    #     vmax = np.std(w) * 3
    #     ax.imshow(w, origin="lower", vmin=-vmax, vmax=vmax)
    #     ax.set_title(f"Scale {i}\nell~{wt.lmids[i]:.0f}")
    # plt.tight_layout(); plt.savefig("wavelet_scales.png", dpi=150)

Choosing a basis
-----------------

.. code-block:: python

    from pixell import enmap, uharm, wavelets, utils

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=0.5 * utils.arcmin,
    )
    uht = uharm.UHT(shape, wcs)

    # ButterTrim with a step ratio of sqrt(2) (finer scale sampling)
    wt_fine = wavelets.WaveletTransform(uht, basis=wavelets.ButterTrim(step=2**0.5))

    # CosineNeedlet peaking at ell = 300, 600, 1200, 2400
    lpeaks = [300, 600, 1200, 2400]
    wt_needlet = wavelets.WaveletTransform(
        uht, basis=wavelets.CosineNeedlet(lpeaks=lpeaks)
    )

    # Restrict to a specific ell range
    wt_restricted = wavelets.WaveletTransform(
        uht, basis=wavelets.ButterTrim(lmin=200, lmax=3000)
    )

Scale-dependent operations
---------------------------

The key advantage of wavelets is the ability to apply *different* operations to
different angular scales. A common use case is scale-dependent noise weighting:

.. code-block:: python

    from pixell import enmap, uharm, wavelets, multimap, utils
    import numpy as np

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=0.5 * utils.arcmin,
    )
    m = enmap.read_map("my_map.fits")

    uht  = uharm.UHT(shape, wcs)
    wt   = wavelets.WaveletTransform(uht)
    wmap = wt.map2wave(m)

    # Measure the variance (noise level) in each wavelet scale
    var_per_scale = multimap.var(wmap)   # shape (nlevel,)

    # Inverse-variance weight each scale
    for i, w in enumerate(wmap.maps):
        if var_per_scale[i] > 0:
            w /= var_per_scale[i]

    # Reconstruct
    m_whitened = wt.wave2map(wmap)

Scale-dependent filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pixell import enmap, uharm, wavelets, utils
    import numpy as np

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=0.5 * utils.arcmin,
    )
    m = enmap.rand_gauss(shape, wcs)

    uht  = uharm.UHT(shape, wcs)
    wt   = wavelets.WaveletTransform(uht)
    wmap = wt.map2wave(m)

    # Suppress small-scale power (high-ell bands)
    for i, w in enumerate(wmap.maps):
        if wt.lmids[i] > 1500:
            w *= 0.1   # strongly suppress

    m_filtered = wt.wave2map(wmap)

    #TODO: add figure -- run code:
    # from pixell import enplot
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # axes[0].imshow(m, origin="lower"); axes[0].set_title("Input")
    # axes[1].imshow(m_filtered, origin="lower"); axes[1].set_title("High-ell suppressed")
    # plt.tight_layout(); plt.savefig("wavelet_filter.png", dpi=150)

Wavelet power spectrum
-----------------------

The variance of each wavelet map is directly related to the angular power spectrum
at that scale. This gives a fast, robust power spectrum estimator:

.. code-block:: python

    from pixell import enmap, uharm, wavelets, multimap, utils
    import numpy as np

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=0.5 * utils.arcmin,
    )
    m = enmap.rand_gauss(shape, wcs)

    uht  = uharm.UHT(shape, wcs)
    wt   = wavelets.WaveletTransform(uht)
    wmap = wt.map2wave(m)

    # Variance per wavelet scale ~ C_ell at lmid
    var_scales = multimap.var(wmap)   # shape (nlevel,)
    ell_mids   = wt.lmids             # shape (nlevel,)

    #TODO: add figure -- run code:
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(7, 4))
    # plt.loglog(ell_mids, var_scales, 'o-', label="Wavelet variance")
    # plt.xlabel(r"$\ell$")
    # plt.ylabel(r"Variance per scale")
    # plt.title("Wavelet power spectrum estimate")
    # plt.grid(True, alpha=0.3); plt.legend()
    # plt.tight_layout(); plt.savefig("wavelet_power.png", dpi=150)

Curved-sky wavelet transforms
-------------------------------

:py:class:`pixell.wavelets.WaveletTransform` works identically for full-sky or
large-patch maps when the UHT is in curved-sky mode:

.. code-block:: python

    from pixell import enmap, uharm, wavelets, curvedsky, powspec, utils

    # Full-sky geometry
    shape, wcs = enmap.geometry2(res=3 * utils.arcmin)

    # Curved-sky UHT with lmax=2000
    uht = uharm.UHT(shape, wcs, mode="curved", lmax=2000)
    wt  = wavelets.WaveletTransform(uht)

    # Simulate a CMB map
    ps  = powspec.read_spectrum("camb_lensedCls.dat", scale=True)
    m   = curvedsky.rand_map(shape, wcs, ps[0:1, 0:1], lmax=2000, seed=1)

    # Wavelet decomposition on the curved sky
    wmap = wt.map2wave(m)
    m_rec = wt.wave2map(wmap)

Haar wavelet transform
-----------------------

:py:class:`pixell.wavelets.HaarTransform` provides a simpler, orthogonal 2D
Haar-like wavelet transform that does not use harmonic space. It is faster and
has no mode leakage, but does not have the harmonic localization of
:py:class:`WaveletTransform`:

.. code-block:: python

    from pixell import enmap, wavelets, utils

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=0.5 * utils.arcmin,
    )
    m = enmap.rand_gauss(shape, wcs)

    ht   = wavelets.HaarTransform(shape, wcs)
    hmap = ht.map2wave(m)
    m_rec = ht.wave2map(hmap)
