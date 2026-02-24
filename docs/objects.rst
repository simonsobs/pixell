Map objects
===========

The central data structure in ``pixell`` is the :py:class:`pixell.enmap.ndmap`. It
is a subclass of :class:`numpy.ndarray` that carries a `World Coordinate System
<https://docs.astropy.org/en/stable/wcs/index.html>`_ (WCS) object alongside the
pixel data. This means that every ``ndmap`` knows where it lives on the sky: slicing,
arithmetic, and I/O all propagate the geometry automatically.

Structure of an ndmap
---------------------

An ``ndmap`` has two parts:

* **The array data**: a standard numpy array of shape ``(..., ny, nx)``.  The last two
  axes are the spatial (dec, RA) axes.  Any leading axes can hold e.g. Stokes
  components, frequency bands, or time samples.
* **The WCS**: an :class:`astropy.wcs.WCS` object that maps the last two pixel axes
  to sky coordinates (see :doc:`geometry <./geometry>` for details).

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    # Create a simple 2-degree patch at 1 arcmin resolution
    shape, wcs = enmap.geometry2(
        pos=np.array([[-1, -1], [1, 1]]) * utils.degree,
        res=1 * utils.arcmin,
    )
    m = enmap.zeros(shape, wcs)

    print(type(m))    # <class 'pixell.enmap.ndmap'>
    print(m.shape)    # (121, 121)  -- (ny, nx)
    print(m.wcs)      # astropy WCS object

A polarized map simply adds a leading Stokes axis:

.. code-block:: python

    # T, Q, U map
    m_pol = enmap.zeros((3,) + shape, wcs)
    print(m_pol.shape)  # (3, 121, 121)

Key properties
--------------

``ndmap`` exposes several convenient properties:

.. code-block:: python

    # Shape of the last two (sky) axes as a (shape, wcs) tuple
    geo_shape, geo_wcs = m.geometry

    # Number of sky pixels (ny * nx)
    print(m.npix)   # 14641

    # Flatten all non-sky axes, keeping the two sky axes
    # Useful for iterating over all components uniformly
    m_4d = enmap.zeros((2, 3) + shape, wcs)
    print(m_4d.preflat.shape)  # (6, ny, nx)

Coordinate methods
------------------

Because an ``ndmap`` knows its geometry, it can convert between pixel indices and
sky coordinates directly.

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    shape, wcs = enmap.geometry2(
        pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
        res=0.5 * utils.arcmin,
    )
    m = enmap.zeros(shape, wcs)

    # Sky position (dec, ra) in radians → pixel index (y, x)
    pos_rad = np.array([[0.0], [0.0]])   # equatorial origin
    pix = m.sky2pix(pos_rad)
    print(pix)  # approximate center of the map

    # Pixel index → sky position
    pos = m.pix2sky(np.array([[0.], [0.]]))  # top-left pixel
    print(np.rad2deg(pos))   # dec and ra in degrees

    # Map of sky positions for every pixel: shape ({dec,ra}, ny, nx)
    posmap = m.posmap()
    dec_map = posmap[0]   # declination of each pixel, in radians
    ra_map  = posmap[1]   # right ascension of each pixel, in radians

    #TODO: add figure -- run code:
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # axes[0].imshow(np.rad2deg(dec_map), origin="lower")
    # axes[0].set_title("Declination map (deg)")
    # axes[1].imshow(np.rad2deg(ra_map), origin="lower")
    # axes[1].set_title("RA map (deg)")
    # plt.tight_layout(); plt.savefig("posmap.png", dpi=150)

Fourier-space coordinate maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alongside the sky-space coordinate maps, ``ndmap`` provides Fourier-space equivalents:

.. code-block:: python

    # 2D Fourier multipole map: shape ({ly, lx}, ny, nx)
    lmap = m.lmap()
    ly_map = lmap[0]
    lx_map = lmap[1]

    # Absolute value of the 2D multipole at each Fourier pixel
    modl = m.modlmap()   # shape (ny, nx)

    # Radial angular-scale map (in radians) from the map center
    modr = m.modrmap()   # shape (ny, nx), units radians

    #TODO: add figure -- run code:
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # axes[0].imshow(modl, origin="lower")
    # axes[0].set_title("|ell| map")
    # axes[1].imshow(np.rad2deg(modr), origin="lower")
    # axes[1].set_title("Angular radius map (deg)")
    # plt.tight_layout(); plt.savefig("lmap.png", dpi=150)

Pixel size and area
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Approximate pixel side lengths in radians (dy, dx)
    dy, dx = m.pixshape()
    print(f"pixel size: {np.rad2deg(dy)*60:.3f} x {np.rad2deg(dx)*60:.3f} arcmin")

    # Pixel area in steradians (scalar, approximate average)
    pix_area = m.pixsize()

    # Per-pixel area map in steradians (varies with declination for CAR)
    area_map = m.pixsizemap()

    # Total map area in steradians
    total_area = m.area()
    print(f"map area: {total_area * (180/np.pi)**2:.2f} deg^2")

Bounding box
^^^^^^^^^^^^

.. code-block:: python

    # [[min_dec, min_ra], [max_dec, max_ra]] in radians
    corners = m.box()
    print(np.rad2deg(corners))

    # Check whether positions lie within the map footprint
    pos_test = np.array([[0.0, 0.01], [0.0, 0.01]])   # two (dec, ra) positions
    inside = m.contains(pos_test)
    print(inside)  # [True, True] or similar

Extracting sub-maps
-------------------

You can extract regions using numpy slice notation or by specifying a sky coordinate box:

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    m = enmap.read_map("my_map.fits")

    # Slice by pixel index (standard numpy) -- WCS updated automatically
    cut = m[..., 100:200, 150:250]

    # Slice by sky coordinate box: [[dec_min, ra_max], [dec_max, ra_min]] in radians
    box = np.array([[-2, 2], [2, -2]]) * utils.degree
    patch = m.submap(box)

.. note::
   In pixell, RA increases to the *left* in the map (as when looking at the sky from
   below). A box ``[[dec_min, ra_max], [dec_max, ra_min]]`` therefore covers
   positive-RA territory.

Downgrading and upgrading
--------------------------

``downgrade`` replaces groups of n×n pixels with their average, reducing the map
resolution:

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    shape, wcs = enmap.geometry2(res=0.5*utils.arcmin,
                                  pos=np.array([[-2,-2],[2,2]])*utils.degree)
    m = enmap.zeros(shape, wcs)

    # Halve the resolution (each output pixel averages 2x2 input pixels)
    m_lo = m.downgrade(2)

    # Upgrade (nearest-neighbour replication)
    m_hi = m_lo.upgrade(2)

    # Custom reduction -- e.g. take the maximum within each block
    m_max = m.downgrade(4, op=np.max)

Resampling
----------

:py:func:`pixell.enmap.resample` changes the pixel count while keeping the same
footprint. The default FFT method preserves the power spectrum up to the Nyquist
frequency of the output:

.. code-block:: python

    from pixell import enmap
    import numpy as np

    m = enmap.read_map("my_map.fits")

    # Resample to half the pixels in each dimension
    new_shape = (m.shape[-2] // 2, m.shape[-1] // 2)
    m_resampled = m.resample(new_shape)

Projecting to a new geometry
-----------------------------

:py:meth:`pixell.enmap.ndmap.project` reprojects a map onto any target geometry using
spline interpolation. Use this when the source and target geometries are not
pixel-compatible:

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    src = enmap.read_map("my_map.fits")

    # Build a target geometry at double the resolution
    shape_out, wcs_out = enmap.geometry2(
        pos=src.box(),
        res=0.5 * src.pixshape()[0],
    )

    m_fine = src.project(shape_out, wcs_out, order=3)

For pixel-compatible geometries (same projection and resolution), use the cheaper
:py:func:`pixell.enmap.extract`:

.. code-block:: python

    m_sub = enmap.extract(src, shape_out, wcs_out)

Inserting a patch into a larger map
-------------------------------------

.. code-block:: python

    from pixell import enmap, utils

    shape_full, wcs_full = enmap.geometry2(res=1.0 * utils.arcmin)
    canvas = enmap.zeros(shape_full, wcs_full)

    patch = enmap.read_map("my_patch.fits")
    enmap.insert(canvas, patch)   # add patch into canvas in-place

Finding the peak position
--------------------------

.. code-block:: python

    # Returns peak position as (dec, ra) in radians
    peak_pos = m.argmax()
    print(np.rad2deg(peak_pos))

    # Minimum position
    min_pos = m.argmin()

Interpolating at arbitrary sky positions
-----------------------------------------

.. code-block:: python

    # Evaluate the map at one or more (dec, ra) positions via spline interpolation
    pos_rad = np.array([[0.0, 0.01], [0.0, 0.01]])  # (dec, ra) per column
    val = m.at(pos_rad, order=3)
    print(val)
