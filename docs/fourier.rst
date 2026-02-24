Fourier analysis
================

Pixell provides tools for flat-sky 2D Fourier analysis directly on
:py:class:`pixell.enmap.ndmap` objects. Because CAR maps have uniform pixel spacing
in angle, the standard 2D FFT is an excellent approximation to the flat-sky
spherical harmonic transform for patches that subtend much less than ~10°. For
large patches or full-sky work see :doc:`harmonic <./harmonic>`.

.. note::
   The flat-sky approximation becomes inaccurate for patches wider than roughly 10°.
   Use :py:mod:`pixell.curvedsky` for large or full-sky maps.

Transforms
----------

:py:func:`pixell.enmap.fft` and :py:func:`pixell.enmap.ifft` perform the 2D FFT
and its inverse:

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    # Build a small test map
    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=1 * utils.arcmin,
    )
    m = enmap.rand_gauss(shape, wcs)   # white-noise map

    # Forward FFT: returns a complex ndmap with the same geometry
    fmap = enmap.fft(m)
    print(fmap.shape)   # (600, 600) -- complex
    print(fmap.dtype)   # complex128

    # Inverse FFT: back to real space
    m_rec = enmap.ifft(fmap).real
    print(np.max(np.abs(m_rec - m)))   # ~1e-13 (numerical noise)

The default normalization is such that the round-trip ``ifft(fft(m)) == m`` holds.
A second normalization, ``normalize="phys"``, additionally divides by ``sqrt(pixsize)``
in the forward direction so that ``sum(|fmap|^2) * dl^2`` approximates the
continuum power spectrum integral:

.. code-block:: python

    fmap_phys = enmap.fft(m, normalize="phys")

Polarization: map2harm / harm2map
-----------------------------------

For T,Q,U polarization maps, :py:func:`pixell.enmap.map2harm` applies the FFT
*and* rotates Q,U into E,B modes in Fourier space. The inverse is
:py:func:`pixell.enmap.harm2map`:

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=1 * utils.arcmin,
    )
    # Simulate a T,Q,U map
    m_pol = enmap.rand_gauss((3,) + shape, wcs)

    # map → Fourier (T, E, B modes)
    f_teb = enmap.map2harm(m_pol, spin=[0, 2])

    # Fourier → map (T, Q, U)
    m_rec = enmap.harm2map(f_teb, spin=[0, 2])

The ``spin`` argument tells the transform which components are spin-0 (temperature)
and which are spin-2 (polarization). The default ``spin=[0, 2]`` is appropriate for
a (T, Q, U) map where ``m[0]`` is T and ``m[1:3]`` are Q and U.

Fourier-space coordinate maps
------------------------------

Several helper functions give the multipole coordinates in Fourier space.  These are
methods on any ``ndmap`` (see :doc:`objects <./objects>`), but can also be called
as module-level functions:

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=1 * utils.arcmin,
    )

    # 2D multipole map: shape ({ly, lx}, ny, nx)
    lmap = enmap.lmap(shape, wcs)

    # |ell| at each Fourier pixel
    modl = enmap.modlmap(shape, wcs)   # shape (ny, nx)

    # 1D ell axes (separable for CAR)
    ly, lx = enmap.laxes(shape, wcs)

    #TODO: add figure -- run code:
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # axes[0].imshow(modl, origin="lower", vmax=3000)
    # axes[0].set_title("|ell| map in Fourier space")
    # fmap = enmap.fft(enmap.rand_gauss(shape, wcs))
    # axes[1].imshow(np.log10(np.abs(fmap)**2 + 1), origin="lower")
    # axes[1].set_title("log10 power in Fourier space")
    # plt.tight_layout(); plt.savefig("fourier_lmap.png", dpi=150)

Isotropic filtering
--------------------

A common task is to apply an azimuthally symmetric filter :math:`F(\ell)` to a map.
The recipe is:

1. FFT the map.
2. Evaluate the filter at each Fourier pixel using ``modlmap``.
3. Multiply.
4. Inverse FFT.

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=1 * utils.arcmin,
    )
    m = enmap.rand_gauss(shape, wcs)

    # --- Gaussian beam smoothing ---
    fwhm = 5 * utils.arcmin                    # beam FWHM
    sigma = fwhm / (8 * np.log(2)) ** 0.5     # sigma in radians
    modl  = enmap.modlmap(shape, wcs)
    beam  = np.exp(-0.5 * modl**2 * sigma**2)

    fmap    = enmap.fft(m)
    m_smooth = enmap.ifft(fmap * beam).real

    #TODO: add figure -- run code:
    # from pixell import enplot
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # axes[0].imshow(m, origin="lower"); axes[0].set_title("Input (white noise)")
    # axes[1].imshow(m_smooth, origin="lower"); axes[1].set_title("Smoothed (5' FWHM beam)")
    # plt.tight_layout(); plt.savefig("beam_smooth.png", dpi=150)

High-pass and low-pass filters work the same way:

.. code-block:: python

    # Band-pass: keep multipoles 500 < ell < 3000
    filt = (modl > 500) & (modl < 3000)
    m_bp = enmap.ifft(enmap.fft(m) * filt).real

    # Wiener-like deconvolution: divide by beam (with regularization)
    beam_inv = beam / (beam**2 + 1e-4)
    m_deconv = enmap.ifft(enmap.fft(m_smooth) * beam_inv).real

2D power spectra
-----------------

To compute the azimuthally averaged 1D power spectrum (for diagnostic purposes):

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=1 * utils.arcmin,
    )
    m = enmap.rand_gauss(shape, wcs)

    # FFT and compute 2D power
    fmap  = enmap.fft(m, normalize="phys")
    p2d   = np.abs(fmap)**2   # 2D power spectrum

    # Bin into 1D using lbin()
    ls, cl = enmap.lbin(p2d, bsize=50)

    #TODO: add figure -- run code:
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(6, 4))
    # plt.loglog(ls, cls)
    # plt.xlabel(r"$\ell$")
    # plt.ylabel(r"$C_\ell$")
    # plt.title("Flat-sky 1D power spectrum")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout(); plt.savefig("cl_flatsky.png", dpi=150)

For cross-spectra between two maps:

.. code-block:: python

    m2 = enmap.rand_gauss(shape, wcs)
    fmap2 = enmap.fft(m2, normalize="phys")

    # Cross-power
    p2d_cross = (fmap * np.conj(fmap2)).real
    ls, cl_cross = enmap.lbin(p2d_cross, bsize=50)

Radial binning in real space
-----------------------------

:py:func:`pixell.enmap.rbin` bins a map by angular radius from a reference point:

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    shape, wcs = enmap.geometry2(
        pos=np.array([[-2, -2], [2, 2]]) * utils.degree,
        res=0.5 * utils.arcmin,
    )
    m = enmap.rand_gauss(shape, wcs)

    # Average in annuli of 1 arcmin width, centered on map center
    rs, profile = enmap.rbin(m, bsize=1 * utils.arcmin)
    print(rs * (180 * 60 / np.pi))    # bin centers in arcmin

Apodization before FFT
-----------------------

Bright edges cause ringing in the FFT. Always apodize before transforming a
real-data patch:

.. code-block:: python

    from pixell import enmap, utils

    m = enmap.read_map("my_patch.fits")

    # Taper the edges over 5 arcmin with a cosine profile
    m_apod = enmap.apod(m, width=5 * utils.arcmin, profile="cos")

    fmap = enmap.fft(m_apod, normalize="phys")

See :doc:`masking <./masking>` for more details on apodization and windowing.
