Fourier Analysis
================

``pixell`` supports flat-sky Fourier analysis of :py:class:`pixell.enmap.ndmap` objects.
Because an ``ndmap`` carries its own geometry, it knows the physical size of its pixels and
the total extent of the map.  This means that the raw FFT output --- which is indexed by
discrete pixel frequencies --- can be directly translated into physical angular wavenumbers
(multipoles).  Everything below assumes you have already imported the relevant modules::

   from pixell import enmap, utils
   import numpy as np


Multipole maps
--------------

Before transforming a map it is useful to know which multipoles correspond to each Fourier
pixel.  ``enmap`` provides three functions for this.

``laxes`` returns the 1-D multipole arrays along the y and x directions:

.. code-block:: python

   >>> shape, wcs = enmap.geometry(pos=np.array([[-5,10],[5,-10]])*utils.degree,
   ...                              res=0.5*utils.arcmin, proj="car")
   >>> ly, lx = enmap.laxes(shape, wcs)
   >>> print(ly.shape, lx.shape)
   (600,) (1200,)

``lmap`` returns a 2-D ``ndmap`` of shape ``(2, ny, nx)`` containing the ``(ly, lx)``
multipole pair for every Fourier pixel:

.. code-block:: python

   >>> lm = enmap.lmap(shape, wcs)
   >>> print(lm.shape)   # (2, ny, nx)
   (2, 600, 1200)
   >>> ly_2d = lm[0]     # multipole along y for each Fourier pixel
   >>> lx_2d = lm[1]     # multipole along x for each Fourier pixel

``modlmap`` gives the radial multipole ``ℓ = √(ly² + lx²)`` for every pixel, which is
the most commonly needed quantity:

.. code-block:: python

   >>> lmod = enmap.modlmap(shape, wcs)
   >>> print(lmod.shape)     # (ny, nx)
   (600, 1200)

These functions can also be called as methods on an existing ``ndmap``::

   lmod = imap.modlmap()


Performing FFTs
---------------

Use :py:func:`pixell.enmap.fft` and :py:func:`pixell.enmap.ifft` to transform between
pixel space and Fourier space.  The transforms are normalised so that ``ifft(fft(m)) == m``
(within floating-point precision).

.. code-block:: python

   >>> kmap = enmap.fft(imap)          # pixel → Fourier
   >>> rmap = enmap.ifft(kmap).real    # Fourier → pixel

``kmap`` is a complex ``ndmap`` with the same shape and WCS as ``imap``.  Because the
input is real-valued, the output is Hermitian-symmetric, but ``enmap.fft`` returns the
full complex array for convenience.

**Physical normalisation.**  By default the normalisation is chosen so that
``ifft(fft(m)) == m``.  If instead you want the power spectrum of ``kmap`` to be
directly comparable to a theory ``C_ℓ`` (corrected for pixel area), pass
``normalize="phys"``::

   kmap_phys = enmap.fft(imap, normalize="phys")


Filtering in Fourier space
--------------------------

A common workflow is to apply an isotropic filter defined in multipole space.  The pattern
is always the same: build a filter on ``modlmap``, multiply in Fourier space, then
transform back.

Gaussian beam smoothing
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   >>> fwhm = 5 * utils.arcmin           # beam FWHM
   >>> sigma = fwhm / (8*np.log(2))**0.5 # convert to sigma
   >>> lmod  = enmap.modlmap(shape, wcs)
   >>> beam  = np.exp(-0.5 * lmod**2 * sigma**2)
   >>> kmap  = enmap.fft(imap)
   >>> omap  = enmap.ifft(kmap * beam).real

High-pass / low-pass filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   >>> lmod = enmap.modlmap(shape, wcs)
   >>> # Keep only ℓ < 3000
   >>> lpf  = (lmod < 3000).astype(float)
   >>> omap = enmap.ifft(enmap.fft(imap) * lpf).real

Matched filter
^^^^^^^^^^^^^^

For a signal with known power spectrum ``C_ℓ^s`` and noise power spectrum ``C_ℓ^n``,
the optimal (Wiener) filter is ``W_ℓ = C_ℓ^s / (C_ℓ^s + C_ℓ^n)``:

.. code-block:: python

   >>> lmod   = enmap.modlmap(shape, wcs)
   >>> # Interpolate 1-D spectra onto the 2-D multipole map
   >>> cls    = np.interp(lmod, np.arange(len(cl_signal)),    cl_signal)
   >>> cln    = np.interp(lmod, np.arange(len(cl_noise)),     cl_noise)
   >>> wiener = cls / (cls + cln)
   >>> omap   = enmap.ifft(enmap.fft(imap) * wiener).real


Flat-sky power spectra
----------------------

A quick-and-dirty diagnostic power spectrum from a single map can be obtained by binning
the square of the Fourier amplitudes.  ``enmap.lbin`` does this automatically:

.. code-block:: python

   >>> kmap = enmap.fft(imap, normalize="phys")
   >>> ps   = np.abs(kmap)**2
   >>> ls, cl = enmap.lbin(ps, bsize=40)

The ``bsize`` argument controls the width of each multipole bin.  ``ls`` contains the
bin centres and ``cl`` the mean power in each bin.  For a multi-component map of shape
``(ncomp, ny, nx)``, ``ps`` will have shape ``(ncomp, ny, nx)`` and each component is
binned independently.

Cross-spectra are formed by multiplying the complex Fourier maps of two maps before
taking the real part and binning::

   kmap1 = enmap.fft(imap1, normalize="phys")
   kmap2 = enmap.fft(imap2, normalize="phys")
   ls, cl_cross = enmap.lbin((kmap1 * kmap2.conj()).real, bsize=40)


Apodisation
-----------

Fourier transforms assume periodic boundary conditions.  When a map has non-zero values
at its edges, the resulting ringing (Gibbs phenomenon) can bias power spectrum estimates.
A standard remedy is to taper the map to zero at its edges before transforming.

:py:func:`pixell.enmap.apod` applies a smooth taper of a given width (in pixels) around
the map boundary:

.. code-block:: python

   >>> imap_apo = enmap.apod(imap, width=60)   # taper over 60 pixels (~30 arcmin at 0.5')
   >>> kmap     = enmap.fft(imap_apo)

The default profile is a raised-cosine (Hann) window.  A linear ramp can be selected
with ``profile="lin"``.  The ``fill`` argument controls what value the taper targets:
``fill="zero"`` (default) tapers towards zero, while ``fill="mean"`` tapers towards the
map mean (useful for maps with a large DC offset).

When computing a power spectrum from an apodised map, divide by the mean squared value
of the taper to correct for the power lost::

   taper     = enmap.apod(enmap.ones(shape, wcs), width=60)
   imap_apo  = imap * taper
   kmap      = enmap.fft(imap_apo, normalize="phys")
   cl_raw    = np.abs(kmap)**2
   # Correct for the taper (W2 factor)
   W2        = np.mean(taper**2)
   cl_corr   = cl_raw / W2


Polarisation: TQU → TEB
------------------------

CMB polarisation maps are conventionally stored as TQU Stokes parameters, but analysis
is typically done in the spin-2 eigenbasis (T, E, B).  :py:func:`pixell.enmap.map2harm`
performs the flat-sky QU → EB rotation and Fourier transform in one step:

.. code-block:: python

   >>> # imap has shape (3, ny, nx) with components [T, Q, U]
   >>> kmap_teb = enmap.map2harm(imap)   # returns complex (3, ny, nx)

The inverse is :py:func:`pixell.enmap.harm2map`::

   omap_tqu = enmap.harm2map(kmap_teb)

Pass ``iau=True`` to use the IAU polarisation convention instead of the COSMO convention.
