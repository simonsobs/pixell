Pixell-provided utilities
=========================

The :py:mod:`pixell.utils` module provides a wide range of utility functions and
physical constants that are used throughout the library and are also useful in
analysis scripts.

Physical constants and unit conversions
-----------------------------------------

A set of physical constants and unit conversion factors are available directly as
module-level attributes. All angular quantities use *radians*:

.. code-block:: python

    from pixell import utils
    import numpy as np

    # Angle conversions (multiply to go from the named unit to radians)
    print(utils.degree)  # pi/180
    print(utils.arcmin)  # pi/(180*60)
    print(utils.arcsec)  # pi/(180*3600)

    # FWHM-to-sigma conversion (sigma = fwhm * utils.fwhm)
    sigma = 1.4 * utils.arcmin * utils.fwhm

    # Physical constants (SI unless noted)
    print(utils.c)       # Speed of light, m/s
    print(utils.h)       # Planck constant, J*s
    print(utils.k)       # Boltzmann constant, J/K
    print(utils.T_cmb)   # CMB temperature, K (2.72548)

    # Astrophysical units
    print(utils.Jy)      # 1 Jansky in W/m^2/Hz (1e-26)
    print(utils.pc)      # 1 parsec in m
    print(utils.AU)      # 1 AU in m

Usage examples:

.. code-block:: python

    from pixell import utils
    import numpy as np

    # Convert a beam FWHM from arcmin to sigma in radians
    fwhm_arcmin = 1.4
    sigma = fwhm_arcmin * utils.arcmin * utils.fwhm
    print(f"sigma = {sigma:.4f} rad = {np.rad2deg(sigma)*60:.4f} arcmin")

    # A 10 degree patch
    patch_rad = 10 * utils.degree

    # Noise level: 10 uK-arcmin in SI
    noise_si = 10 * utils.arcmin   # uK*rad

CMB unit conversions
---------------------

:py:func:`pixell.utils.dplanck` gives the derivative of the Planck spectrum with
respect to temperature, in units of Jy/sr/K. This converts between CMB temperature
fluctuations (in uK) and specific intensity (in Jy/sr or mJy/sr):

.. code-block:: python

    from pixell import utils

    # Conversion factor at 150 GHz: uK → mJy/sr
    fconv = utils.dplanck(150e9, utils.T_cmb) / 1e3   # Jy/sr/K → mJy/sr/uK
    print(f"1 uK_CMB = {fconv:.4f} mJy/sr at 150 GHz")

    # Convert a 100 uK source to mJy/sr
    T_uK  = 100.0
    I_mJy = T_uK * fconv
    print(f"{T_uK} uK = {I_mJy:.2f} mJy/sr")

    # tSZ spectrum (Compton-y → Jy/sr)
    y_factor = utils.tsz_spectrum(150e9) * 1e26   # Jy/sr per unit y
    print(f"tSZ: 1 unit y = {y_factor:.1f} Jy/sr at 150 GHz")

Multipole / angle conversions
-------------------------------

.. code-block:: python

    from pixell import utils
    import numpy as np

    # Rough angular scale corresponding to multipole ell
    ell  = 3000
    ang  = utils.l2ang(ell)
    print(f"ell={ell} → theta ~ {np.rad2deg(ang)*60:.1f} arcmin")

    # Inverse: multipole corresponding to an angular scale
    theta = 1.0 * utils.arcmin
    ell_out = utils.ang2l(theta)
    print(f"theta=1 arcmin → ell ~ {ell_out:.0f}")

Array and numerical utilities
-------------------------------

.. code-block:: python

    from pixell import utils
    import numpy as np

    # Unwind angles to avoid 2π jumps (like np.unwrap but more flexible)
    angles = np.array([0.0, 1.0, 6.1, 0.3])
    unwound = utils.unwind(angles)

    # Atleast_3d: ensure array has at least 3 dimensions (like np.atleast_3d)
    arr = np.array([1.0, 2.0, 3.0])
    arr3d = utils.atleast_3d(arr)   # shape (1, 1, 3)

    # Numerical derivative using complex step (accurate to O(eps^2))
    # D(f)(x) returns f'(x) using a single function evaluation
    deriv_sin = utils.D(np.sin)(0.0)   # should be cos(0) = 1.0
    print(f"d/dx sin(x)|_0 = {deriv_sin:.6f}")

    # Weighted median
    vals = np.array([1.0, 2.0, 3.0, 10.0])
    ivar = np.array([1.0, 1.0, 1.0, 0.001])  # outlier has low weight
    wmed = utils.weighted_median(vals, ivar)
    print(f"weighted median = {wmed:.2f}")  # ~2.0, not 10.0

I/O utilities
--------------

.. code-block:: python

    import os
    from pixell import utils

    # Ensure a directory exists (like mkdir -p)
    utils.mkdir("output/maps")

    # Touch a file (create it if it doesn't exist, update its timestamp)
    utils.touch("output/done.flag")

    # Read lines from a file (returns an iterator)
    for line in utils.lines("my_catalog.txt"):
        print(line.strip())

The FFT module
--------------

:py:mod:`pixell.fft` is a thin wrapper around multiple FFT backends (numpy,
pyfftw, ducc0). It is used internally by :py:func:`pixell.enmap.fft` and
:py:func:`pixell.enmap.ifft`, but can also be used directly:

.. code-block:: python

    from pixell import fft as pfft
    import numpy as np

    # Check which backend is active
    print(pfft.engine)   # "fftw", "ducc", or "numpy"

    # Direct FFT (axes = last two)
    arr = np.random.randn(128, 128)
    out = np.zeros_like(arr, dtype=complex)
    pfft.fft(arr, out, axes=[-2, -1])

    # Inverse FFT
    arr_rec = np.zeros_like(arr)
    pfft.ifft(out, arr_rec, axes=[-2, -1], normalize=True)

    # Find the next FFT-friendly array length
    n_good = pfft.fft_len(257, direction="above", factors=[2, 3, 5])
    print(n_good)   # 270 (= 2 * 3^3 * 5)

The ``fft_len`` function is useful for padding arrays to lengths that avoid slow
FFT sizes:

.. code-block:: python

    from pixell import fft as pfft

    # When extracting thumbnails, round up to FFT-friendly size
    raw_size = 311
    padded   = pfft.fft_len(raw_size, direction="above", factors=[2, 3, 5])
    print(f"Pad {raw_size} → {padded}")   # e.g. 320

The powspec module
-------------------

:py:mod:`pixell.powspec` provides I/O for angular power spectra:

.. code-block:: python

    from pixell import powspec

    # Read a CAMB scalar spectrum file
    # scale=True converts from D_ell = ell*(ell+1)*C_ell/(2*pi) to C_ell
    ps = powspec.read_spectrum("camb_scalCls.dat", scale=True)
    print(ps.shape)  # (ncomp, ncomp, lmax+1)

    # Expand a symmetric 1D spectrum to a full covariance matrix
    cl_TT = ps[0, 0]   # TT spectrum
    cov   = powspec.sym_expand(cl_TT[None], scheme="diag")  # (1, 1, lmax+1)

The Bunch class
----------------

:py:class:`pixell.bunch.Bunch` is a simple attribute-access dictionary, useful for
collecting outputs from functions:

.. code-block:: python

    from pixell.bunch import Bunch

    result = Bunch(flux=42.3, snr=7.1, pos=[0.0, 0.1])
    print(result.flux)   # 42.3
    print(result.snr)    # 7.1

    # Can also be used like a dict
    result["sigma"] = 5.9
    print(result.sigma)  # 5.9

MPI utilities
--------------

Pixell has optional MPI support through :py:mod:`pixell.mpi`. When MPI is
available, it provides a simple interface for distributing work across ranks:

.. code-block:: python

    from pixell import mpi

    comm  = mpi.COMM_WORLD
    rank  = comm.rank
    nrank = comm.size

    # Distribute a list of files across MPI ranks
    files = [f"map_{i:04d}.fits" for i in range(100)]
    my_files = files[rank::nrank]   # each rank gets every nrank-th file

    for fname in my_files:
        # process fname on this rank
        pass

    # Synchronize
    comm.Barrier()
