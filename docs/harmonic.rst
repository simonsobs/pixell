Spherical harmonic analysis
===========================

For large patches or full-sky analysis the flat-sky Fourier approximation breaks down
(see :doc:`fourier <./fourier>`). Pixell wraps the `ducc0
<https://gitlab.mpcdf.mpg.de/mtr/ducc>`_ library to provide fast and accurate
*curved-sky* spherical harmonic transforms (SHTs) through the
:py:mod:`pixell.curvedsky` module. These transforms work directly on
:py:class:`pixell.enmap.ndmap` objects in CAR pixelization.

.. note::
   For the fastest and most accurate ``map2alm`` transform, use a geometry produced
   by :py:func:`pixell.enmap.geometry2` with the default Fejer1 pixelization. See
   :doc:`geometry <./geometry>` for details.

Spherical harmonic coefficients (alms)
----------------------------------------

Spherical harmonic coefficients are stored as 1D or 2D numpy arrays in the same
triangular layout used by ``healpy``. The layout is indexed by a single integer
``idx = m*(2*lmax+1-m)//2 + l``, where ``l`` is the multipole and ``m`` its
azimuthal order. For a set of *n* components the shape is ``(n, nalm)`` where
``nalm = (lmax+1)*(lmax+2)//2``.

The :py:class:`pixell.curvedsky.alm_info` class describes the layout and provides
utilities for working with alms:

.. code-block:: python

    from pixell import curvedsky
    import numpy as np

    lmax  = 3000
    ainfo = curvedsky.alm_info(lmax=lmax)
    print(ainfo.lmax)   # 3000
    print(ainfo.nalm)   # (lmax+1)*(lmax+2)//2

Map to alm (analysis)
-----------------------

:py:func:`pixell.curvedsky.map2alm` decomposes an enmap into spherical harmonic
coefficients:

.. code-block:: python

    from pixell import enmap, curvedsky, utils
    import numpy as np

    # Build a full-sky Fejer1 geometry at 1 arcmin resolution
    shape, wcs = enmap.geometry2(res=1 * utils.arcmin)

    # Fill with a Gaussian random field
    m = enmap.rand_gauss(shape, wcs)

    # Transform to harmonic space
    lmax = 5000
    alm  = curvedsky.map2alm(m, lmax=lmax)
    print(alm.shape)   # (nalm,) for a single component

    # For a polarized map (T, Q, U):
    m_pol = enmap.rand_gauss((3,) + shape, wcs)
    alm_pol = curvedsky.map2alm(m_pol, lmax=lmax, spin=[0, 2])
    print(alm_pol.shape)   # (3, nalm): T, E, B alms

The ``spin=[0, 2]`` argument tells the transform that the first component is
spin-0 (temperature) and the remaining two are spin-2 (polarization Q, U → E, B).

Iterative refinement
^^^^^^^^^^^^^^^^^^^^^

By default ``map2alm`` uses pixel-area quadrature weights. For small patches or
maps that do not have exact quadrature weights available you can use Jacobi
iterations to improve accuracy:

.. code-block:: python

    alm = curvedsky.map2alm(m, lmax=lmax, niter=3)

Each iteration adds a correction that typically reduces the residual by an order
of magnitude.

Alm to map (synthesis)
------------------------

:py:func:`pixell.curvedsky.alm2map` synthesizes a map from spherical harmonic
coefficients:

.. code-block:: python

    from pixell import enmap, curvedsky, utils
    import numpy as np

    shape, wcs = enmap.geometry2(res=1 * utils.arcmin)
    m_out = enmap.zeros(shape, wcs)

    # alm → map (overwrites m_out in-place, also returns it)
    curvedsky.alm2map(alm, m_out)

    # Or using a pre-allocated output for polarization:
    m_pol_out = enmap.zeros((3,) + shape, wcs)
    curvedsky.alm2map(alm_pol, m_pol_out, spin=[0, 2])

Round-trip check:

.. code-block:: python

    alm   = curvedsky.map2alm(m, lmax=lmax)
    m_rec = enmap.zeros(shape, wcs)
    curvedsky.alm2map(alm, m_rec)
    print(np.max(np.abs(m_rec - m)))   # small residual

Angular power spectra
----------------------

The angular power spectrum :math:`C_\ell` measures the variance of the field as a
function of multipole. Use :py:meth:`pixell.curvedsky.alm_info.alm2cl` to compute
it from alms:

.. code-block:: python

    from pixell import enmap, curvedsky, utils
    import numpy as np

    shape, wcs = enmap.geometry2(res=1 * utils.arcmin)
    m = enmap.rand_gauss(shape, wcs)

    lmax  = 3000
    ainfo = curvedsky.alm_info(lmax=lmax)
    alm   = curvedsky.map2alm(m, lmax=lmax)

    # Auto-spectrum
    cl = ainfo.alm2cl(alm, alm)    # shape (lmax+1,)

    # Cross-spectrum between two maps
    m2   = enmap.rand_gauss(shape, wcs)
    alm2 = curvedsky.map2alm(m2, lmax=lmax)
    cl_cross = ainfo.alm2cl(alm, alm2)

    #TODO: add figure -- run code:
    # import matplotlib.pyplot as plt
    # ells = np.arange(lmax + 1)
    # plt.figure(figsize=(7, 4))
    # plt.loglog(ells[2:], cl[2:], label="auto")
    # plt.xlabel(r"$\ell$"); plt.ylabel(r"$C_\ell$")
    # plt.title("Angular power spectrum")
    # plt.legend(); plt.grid(True, alpha=0.3)
    # plt.tight_layout(); plt.savefig("cl_curved.png", dpi=150)

Filtering in harmonic space
-----------------------------

To apply a multipole filter :math:`F(\ell)` to a map via SHTs:

.. code-block:: python

    from pixell import enmap, curvedsky, utils
    import numpy as np

    shape, wcs = enmap.geometry2(res=1 * utils.arcmin)
    m = enmap.rand_gauss(shape, wcs)

    lmax  = 3000
    ainfo = curvedsky.alm_info(lmax=lmax)
    alm   = curvedsky.map2alm(m, lmax=lmax)

    # Gaussian beam filter: B_ell = exp(-ell*(ell+1)*sigma^2 / 2)
    fwhm  = 5 * utils.arcmin
    sigma = fwhm / (8 * np.log(2))**0.5
    ells  = np.arange(lmax + 1)
    beam  = np.exp(-0.5 * ells * (ells + 1) * sigma**2)

    # Multiply alm by the filter (lmul broadcasts over m)
    alm_filtered = ainfo.lmul(alm, beam)

    # Back to map
    m_smooth = enmap.zeros(shape, wcs)
    curvedsky.alm2map(alm_filtered, m_smooth)

    #TODO: add figure -- run code:
    # from pixell import enplot
    # fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    # axes[0].imshow(m[500:700, 500:700], origin="lower"); axes[0].set_title("Input")
    # axes[1].imshow(m_smooth[500:700, 500:700], origin="lower")
    # axes[1].set_title("Beam-smoothed (5' FWHM)")
    # plt.tight_layout(); plt.savefig("beam_smooth_curved.png", dpi=150)

The alm coefficients are compatible with ``healpy``. You can therefore use
``healpy.almxfl`` as an alternative to ``ainfo.lmul``:

.. code-block:: python

    import healpy
    alm_filtered = healpy.almxfl(alm, beam)

Generating random alms
-----------------------

:py:func:`pixell.curvedsky.rand_alm` generates Gaussian-random alms with a given
power spectrum, using less memory than ``healpy.synalm``:

.. code-block:: python

    from pixell import curvedsky, powspec, utils
    import numpy as np

    # Load a theory power spectrum (e.g. from CAMB)
    # ps shape: (lmax+1,) for a single component, or (ncomp, ncomp, lmax+1)
    ps = powspec.read_spectrum("camb_lensedCls.dat", scale=True)

    lmax = 3000
    alm  = curvedsky.rand_alm(ps, lmax=lmax, seed=42)

Generating random curved-sky maps
-----------------------------------

:py:func:`pixell.curvedsky.rand_map` combines ``rand_alm`` and ``alm2map`` in one call:

.. code-block:: python

    from pixell import enmap, curvedsky, powspec, utils
    import numpy as np

    shape, wcs = enmap.geometry2(res=1 * utils.arcmin)

    ps  = powspec.read_spectrum("camb_lensedCls.dat", scale=True)
    sim = curvedsky.rand_map(shape, wcs, ps, lmax=3000, seed=1)

    #TODO: add figure -- run code:
    # from pixell import enplot
    # plot = enplot.plot(sim, colorbar=True, range="250")
    # enplot.write("rand_curved", plot)

Evaluating alms at arbitrary positions
----------------------------------------

For non-uniform point sets (e.g. source positions), use
:py:func:`pixell.curvedsky.alm2map_pos`:

.. code-block:: python

    from pixell import curvedsky
    import numpy as np

    # Positions: [[dec1, dec2, ...], [ra1, ra2, ...]] in radians
    pos = np.array([[0.0, 0.1, -0.1], [0.0, 0.05, 0.1]])
    vals = curvedsky.alm2map_pos(alm, pos=pos, ainfo=ainfo)
    print(vals.shape)   # (3,) -- one value per position

Converting between HEALPix and alms
-------------------------------------

The alm layout is identical to ``healpy``'s convention, so you can freely mix
pixell SHTs with HEALPix operations:

.. code-block:: python

    import healpy, numpy as np
    from pixell import curvedsky, enmap, utils

    # HEALPix map → alms (using healpy)
    hp_map  = healpy.read_map("my_healpix_map.fits")
    alm_hp  = healpy.map2alm(hp_map, lmax=3000)

    # alms → CAR enmap (using pixell)
    shape, wcs = enmap.geometry2(res=1 * utils.arcmin)
    m_car = enmap.zeros(shape, wcs)
    curvedsky.alm2map(alm_hp, m_car)

See also :py:func:`pixell.reproject.healpix2map` and
:py:func:`pixell.reproject.map2healpix` for direct map-to-map reprojection
without going through alms.
