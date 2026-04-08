Spherical Harmonic Analysis
===========================

The :py:mod:`pixell.curvedsky` module provides spherical harmonic transforms (SHTs)
that correctly account for the curvature of the sky.  Unlike the flat-sky Fourier
transforms in :doc:`fourier`, these operate on the full curved sky and produce
spherical harmonic coefficients (alms) that are compatible with the HEALPix convention
used by ``healpy``.

Everything below assumes::

   from pixell import enmap, curvedsky, utils
   import numpy as np


The alm array convention
------------------------

Spherical harmonic coefficients are complex arrays whose layout follows the HEALPix
(``healpy``) triangular scheme.  For a maximum multipole ``lmax``, the number of
coefficients per component is ``(lmax+1)*(lmax+2)//2``.

The ``alm_info`` class
^^^^^^^^^^^^^^^^^^^^^^

:py:class:`pixell.curvedsky.alm_info` stores layout information for an alm array and
provides utilities for working with it.  Most functions in ``curvedsky`` create one
internally, but you can create one explicitly if you want to reuse the layout metadata
across many operations:

.. code-block:: python

   >>> ainfo = curvedsky.alm_info(lmax=3000)
   >>> print(ainfo.lmax, ainfo.nelem)
   3000 4504501

The ``ainfo`` object can then be passed to ``map2alm`` or ``alm2map`` to skip
recomputing the layout each call.


Synthesis: alm → map
---------------------

:py:func:`pixell.curvedsky.alm2map` takes an alm array and writes the corresponding
sky map into an existing ``enmap``.  The map must be allocated first:

.. code-block:: python

   >>> shape, wcs = enmap.fullsky_geometry(res=1.5*utils.arcmin)   # full-sky Fejer1 geometry
   >>> omap = enmap.zeros((3,)+shape[-2:], wcs)             # TQU map
   >>> curvedsky.alm2map(alm, omap, spin=[0, 2])

The ``spin`` argument tells the transform how to pair up the component axis:
``0`` means a scalar (intensity) transform consuming one component, while ``2`` means a
spin-2 (polarisation) transform consuming the next two.  The default ``spin=[0, 2]``
handles the standard T, Q, U layout.

For a temperature-only map::

   omap = enmap.zeros(shape[-2:], wcs)
   curvedsky.alm2map(alm_T, omap, spin=[0])

The function overwrites ``omap`` in-place and also returns it.  To work on a copy
without modifying the original pass ``copy=True``.


Analysis: map → alm
--------------------

:py:func:`pixell.curvedsky.map2alm` decomposes an ``enmap`` into spherical harmonic
coefficients.

.. code-block:: python

   >>> lmax = 3000
   >>> alm  = curvedsky.map2alm(imap, lmax=lmax, spin=[0, 2])

For accurate results the input map should use a Fejer1 (or other quadrature-compatible)
geometry.  See :doc:`geometry` for how to construct one with
:py:func:`pixell.enmap.fullsky_geometry`.

``map2alm`` returns an alm array of shape ``(..., ncomp, nelem)`` matching the leading
dimensions of the map.

Choosing a transform method
----------------------------

Both ``alm2map`` and ``map2alm`` accept a ``method`` argument.  By default
(``method="auto"``), the best available method is chosen based on the map geometry:

* ``"2d"`` — fastest; requires a full-sky (or paddable) Fejer1, Clenshaw-Curtis or
  similar CAR geometry.
* ``"cyl"`` — works for any cylindrical projection (CAR, CEA, …) with an integer number
  of pixels around the full sky.
* ``"general"`` — works for any pixelisation, including tangent-plane patches, but is
  significantly slower and uses more memory.

You can query which method will be used without running the transform::

   method = curvedsky.get_method(shape, wcs)
   print(method)   # e.g. "2d"

If your workflow involves many transforms on the same geometry, you can precompute the
pixel layout information and pass it via ``locinfo`` (general method) to avoid
recomputing it each time.


Computing power spectra
-----------------------

Given an alm array, ``alm_info.alm2cl`` computes cross power spectra.  To get the
auto-spectrum of a single component:

.. code-block:: python

   >>> ainfo = curvedsky.alm_info(lmax=3000)
   >>> cl    = ainfo.alm2cl(alm)          # shape (nl,)
   >>> ell   = np.arange(ainfo.lmax+1)

For the full TEB×TEB matrix of a three-component alm with shape ``(3, nelem)``:

.. code-block:: python

   >>> cl_mat = ainfo.alm2cl(alm[:, None, :], alm[None, :, :])
   >>> # cl_mat has shape (3, 3, nl): TT, TE, TB, ET, EE, EB, BT, BE, BB

Cross-spectra between two maps are formed the same way::

   alm1 = curvedsky.map2alm(imap1, lmax=3000)
   alm2 = curvedsky.map2alm(imap2, lmax=3000)
   cl_cross = ainfo.alm2cl(alm1, alm2)


Applying a filter in harmonic space
-------------------------------------

To apply an isotropic filter to a map (e.g. beam deconvolution or smoothing), multiply
the alm by the filter values at each ℓ using ``alm_info.lmul``:

.. code-block:: python

   >>> ainfo  = curvedsky.alm_info(lmax=3000)
   >>> ell    = np.arange(ainfo.lmax+1)
   >>> # Smooth with a 5-arcmin Gaussian beam
   >>> sigma  = 5*utils.arcmin / (8*np.log(2))**0.5
   >>> bl     = np.exp(-0.5 * ell*(ell+1) * sigma**2)
   >>> alm_sm = ainfo.lmul(alm, bl)

Because ``lmul`` expects an array of length ``lmax+1``, you can use any ℓ-dependent
filter in this way.  Deconvolving a beam is the same thing with ``1/bl``; be careful
to avoid division by zero at high ℓ where the beam is negligible.


Generating random realisations
--------------------------------

``curvedsky`` provides two functions for generating Gaussian random fields.

``rand_alm`` — alm coefficients from a power spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:func:`pixell.curvedsky.rand_alm` draws random alm coefficients consistent with a
given power spectrum.  The spectrum can be a 1-D array ``C_ℓ`` for a single component,
or a 2-D symmetric array ``C_ℓ[ncomp, ncomp, nl]`` for correlated components:

.. code-block:: python

   >>> cl   = np.zeros(3001)
   >>> cl[2:] = 1e-4 / (np.arange(2, 3001)*(np.arange(2, 3001)+1))
   >>> alm  = curvedsky.rand_alm(cl, lmax=3000)

Pass a ``seed`` for reproducibility::

   alm = curvedsky.rand_alm(cl, lmax=3000, seed=42)

``rand_alm_healpy`` — alm coefficients via healpy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                          
                                                                  
:py:func:`pixell.curvedsky.rand_alm_healpy` is a thin wrapper around
``healpy.synalm``.  It accepts the same spectrum formats as ``rand_alm``
(1-D ``[nl]``, 2-D compressed ``[nspec, nl]``, or 3-D ``[ncomp, ncomp, nl]``)
and returns alm coefficients in the standard HEALPix m-major ordering.

Unlike ``rand_alm``, this function generates random numbers in healpy's native
ordering, so low-res and high-res realisations with the same seed will **not**
agree on large scales.  Use ``rand_alm`` when you need that property
(e.g. for multi-resolution consistency checks).  ``rand_alm_healpy`` is used
internally by ``rand_map``.

``rand_map`` — pixel-space map from a power spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:func:`pixell.curvedsky.rand_map` combines ``rand_alm_healpy`` and ``alm2map`` into one
convenience call that returns a pixel-space ``ndmap`` directly:

.. code-block:: python

   >>> shape, wcs = enmap.fullsky_geometry(res=1.5*utils.arcmin)
   >>> # CMB-like TQU power spectra (shape [3, 3, nl] or [nl] for scalar)
   >>> sim = curvedsky.rand_map((3,)+shape[-2:], wcs, ps, lmax=3000, seed=0)


Working with partial-sky maps
------------------------------

The "2d" and "cyl" methods temporarily pad partial-sky maps to the full sky before
transforming, so you can pass any CAR sub-map without any extra setup:

.. code-block:: python

   >>> box   = np.array([[-20, 40], [20, -40]]) * utils.degree
   >>> shape, wcs = enmap.geometry(pos=box, res=1.5*utils.arcmin)
   >>> omap  = enmap.zeros((3,)+shape[-2:], wcs)
   >>> curvedsky.alm2map(alm, omap, spin=[0, 2])   # pads internally, no user action needed

For maps that are not cylindrical (e.g. tangent-plane patches), ``method="general"``
is selected automatically.  This is accurate but slower; it is most useful for small
cutouts or thumbnails::

   shape, wcs = enmap.thumbnail_geometry(r=30*utils.arcmin, res=0.5*utils.arcmin)
   omap = enmap.zeros(shape, wcs)
   curvedsky.alm2map(alm_T, omap, spin=[0])   # uses "general" automatically


Interfacing with HEALPix
-------------------------

Because the alm convention matches ``healpy``, you can freely mix ``curvedsky`` and
``healpy`` operations.  For example, compute alms from a ``healpy`` map and project
onto a CAR grid:

.. code-block:: python

   >>> import healpy as hp
   >>> alm_hp = hp.map2alm(healpix_map, lmax=3000)   # standard healpy
   >>> shape, wcs = enmap.fullsky_geometry(res=1.5*utils.arcmin)
   >>> omap = enmap.zeros(shape, wcs)
   >>> curvedsky.alm2map(alm_hp, omap, spin=[0])

Going the other direction::

   alm = curvedsky.map2alm(imap, lmax=3000)
   cl  = hp.alm2cl(alm)   # works directly
