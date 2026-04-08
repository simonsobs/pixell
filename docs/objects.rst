Map Objects
===========

The central data structure in ``pixell`` is :py:class:`pixell.enmap.ndmap`.  It combines
a numpy array with an ``astropy.wcs.WCS`` object so that every pixel knows its
position on the sky.  Most operations that work on numpy arrays also work on
``ndmap`` objects, and the WCS is automatically updated when you slice or crop a map.

Everything below assumes::

   from pixell import enmap, utils
   import numpy as np


Creating maps
-------------

The simplest way to get an ``ndmap`` is to read one from disk (see :doc:`loading`) or
generate a random realisation (see :doc:`harmonic`).  You can also create empty or
constant maps the same way you would with numpy:

.. code-block:: python

   >>> shape, wcs = enmap.geometry2(res=0.5*utils.arcmin)        # full-sky Fejer1
   >>> imap = enmap.zeros((3,) + shape[-2:], wcs)                # TQU, all zeros
   >>> imap = enmap.empty(shape, wcs, dtype=np.float32)          # uninitialised
   >>> imap = enmap.ones(shape, wcs)
   >>> imap = enmap.full(shape, wcs, 42.0)                       # fill with a constant

For a small patch::

   box   = np.array([[-5, 10], [5, -10]]) * utils.degree
   shape, wcs = enmap.geometry2(pos=box, res=0.5*utils.arcmin)
   imap  = enmap.zeros(shape, wcs)

To wrap an existing numpy array in an ``ndmap``::

   arr  = np.random.randn(*shape)
   imap = enmap.enmap(arr, wcs)     # copies by default (pass copy=False to avoid)
   imap = enmap.ndmap(arr, wcs)     # zero-copy view

If a function strips the WCS from an ``ndmap`` and returns a plain numpy array,
you can re-attach it::

   omap = enmap.samewcs(some_result, imap)   # borrow WCS from imap


numpy operations
----------------

``ndmap`` inherits every numpy operation.  Arithmetic, broadcasting, and ufuncs all
work as you would expect, and the WCS is carried through:

.. code-block:: python

   >>> # Scale a map
   >>> omap = imap * 1e6
   >>> # Add two maps
   >>> omap = imap1 + imap2
   >>> # Apply a per-pixel weight map
   >>> omap = imap * weight_map

For operations that reduce or change the shape of the last two axes (e.g.
``np.sum(imap, axis=-1)``), the result will no longer have a meaningful WCS and
is returned as a plain numpy array.

Multi-component maps
^^^^^^^^^^^^^^^^^^^^

Many pixell functions work with 3-D maps of shape ``(ncomp, ny, nx)``.  CMB maps
conventionally store temperature and polarisation Stokes parameters T, Q, U as
three components along the leading axis.  You can address individual components
by indexing::

   T_map = imap[0]    # shape (ny, nx)
   Q_map = imap[1]
   U_map = imap[2]

The ``preflat`` property collapses any number of leading dimensions into one,
making it easy to iterate::

   for comp in imap.preflat:
       print(comp.shape)   # each (ny, nx) ndmap


Slicing and selecting regions
------------------------------

Pixel slicing works exactly as with numpy.  The WCS is automatically updated so
that coordinate functions remain correct on the result:

.. code-block:: python

   >>> sub = imap[100:200, 300:500]   # pixel slice
   >>> print(sub.shape)
   (100, 200)

To select by sky coordinates, use :py:meth:`pixell.enmap.ndmap.submap` with a
bounding box in radians.  The box is ``[[dec_from, ra_from], [dec_to, ra_to]]``:

.. code-block:: python

   >>> box = np.array([[-5, 10], [5, -10]]) * utils.degree
   >>> sub = imap.submap(box)

``submap`` accepts partial pixels at the boundary; use the ``mode`` argument to
control rounding (``"floor"``, ``"ceil"``, ``"inclusive"``, ``"exclusive"``).

:py:func:`pixell.enmap.extract` copies part of a map into a pre-allocated output
map of a known geometry.  This is the most efficient way to cut a sub-region when
you already know the target geometry::

   sub = enmap.extract(imap, sub_shape, sub_wcs)


Geometry and coordinate information
-------------------------------------

The geometry of a map is described by its ``shape`` and ``wcs``.  You can inspect
these directly or use the convenience property ``geometry``::

   shape, wcs = imap.geometry

**Sky ↔ pixel conversions.**
:py:meth:`~pixell.enmap.ndmap.sky2pix` and :py:meth:`~pixell.enmap.ndmap.pix2sky`
convert between sky coordinates (dec, RA in radians) and pixel coordinates
(y, x).  Both accept arrays of coordinates for vectorised use:

.. code-block:: python

   >>> pos  = np.array([0.1, -0.3])    # (dec, ra) in radians
   >>> pix  = imap.sky2pix(pos)        # fractional (y, x) pixel
   >>> pos2 = imap.pix2sky(pix)

For many positions at once, pass an array with shape ``(2, N)`` where row 0 is
declination and row 1 is RA.

**Position and pixel maps.**
:py:meth:`~pixell.enmap.ndmap.posmap` returns a ``(2, ny, nx)`` map containing
the ``(dec, ra)`` sky coordinates of every pixel:

.. code-block:: python

   >>> pmap = imap.posmap()
   >>> dec  = pmap[0]
   >>> ra   = pmap[1]

:py:meth:`~pixell.enmap.ndmap.pixmap` returns the integer pixel index ``(y, x)``
of every pixel (useful for building masks).

:py:meth:`~pixell.enmap.ndmap.modrmap` returns the angular distance of every
pixel from a reference point, defaulting to the map centre::

   dist = imap.modrmap()                         # distance from centre
   dist = imap.modrmap(ref=[0.1, -0.3])          # distance from (dec, ra)

**Pixel area.**
:py:meth:`~pixell.enmap.ndmap.pixsizemap` returns the solid angle (in steradians)
of every pixel:

.. code-block:: python

   >>> pixarea = imap.pixsizemap()

For maps near the equator, pixel area varies by less than a percent.
Near the poles the variation can be substantial.

**Map extent and area.**  ``imap.area()`` returns the total solid angle in
steradians.  ``imap.extent()`` returns the approximate angular size along the
two pixel axes.


Resampling
----------

Downgrading (averaging) and upgrading (interpolating) change the pixel resolution
while keeping the geometry valid.

:py:func:`pixell.enmap.downgrade` replaces each ``n×n`` block of pixels with their
mean (or another reduction operator):

.. code-block:: python

   >>> lo_res = enmap.downgrade(imap, 4)        # 4× coarser resolution

:py:func:`pixell.enmap.upgrade` interpolates the map to a higher resolution:

.. code-block:: python

   >>> hi_res = enmap.upgrade(imap, 2)          # 2× finer resolution

For non-integer rescaling, :py:func:`pixell.enmap.resample` uses either FFT or
spline interpolation::

   omap = enmap.resample(imap, (new_ny, new_nx))


Reprojecting to a different geometry
--------------------------------------

When two maps share pixel-compatible geometries (see :doc:`geometry`), you can
copy pixels between them with zero interpolation cost using
:py:func:`pixell.enmap.extract`.  For geometries that are *not* compatible,
:py:func:`pixell.enmap.project` re-samples the map onto an arbitrary new geometry
using spline interpolation:

.. code-block:: python

   >>> target_shape, target_wcs = enmap.geometry2(pos=box, res=1.0*utils.arcmin)
   >>> omap = enmap.project(imap, target_shape, target_wcs)

The ``order`` argument controls the spline order (0 = nearest-neighbour, 1 = linear,
3 = cubic; default 3).  ``mode`` can also be ``"nearest"`` for a fast nearest-pixel
look-up.

:py:func:`pixell.enmap.at` evaluates a map at a list of arbitrary sky positions
(interpolating between pixels) and returns a 1-D array of values — useful for
extracting profiles or sampling at catalogue positions:

.. code-block:: python

   >>> pos   = np.array([[dec1, dec2, ...], [ra1, ra2, ...]])   # (2, N)
   >>> vals  = enmap.at(imap, pos)


Inserting one map into another
--------------------------------

:py:func:`pixell.enmap.insert` adds the values from a smaller map into a larger
one, respecting the WCS:

.. code-block:: python

   >>> enmap.insert(large_map, small_map)

The ``op`` argument controls how pixels are combined.  The default is to overwrite;
use ``op=lambda a,b: a+b`` to accumulate instead.

:py:func:`pixell.enmap.insert_at` does the same at explicit pixel coordinates
rather than using WCS matching::

   enmap.insert_at(omap, pix, stamp)


Writing maps to disk
---------------------

Maps can be written to FITS or HDF5 format:

.. code-block:: python

   >>> enmap.write_map("output.fits", imap)
   >>> enmap.write_map("output.hdf", imap)      # format inferred from extension

The ``ndmap`` method form is equivalent::

   imap.write("output.fits")

The geometry alone (without the data) can be saved and re-loaded independently::

   enmap.write_map_geometry("geometry.fits", shape, wcs)
   shape2, wcs2 = enmap.read_map_geometry("geometry.fits")
