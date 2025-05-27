Sky Geometry
============

An :py:class:`pixell.enmap.ndmap` represents one or more images of the sky, with
a pixelization described by a "geometry" represented as a ``(shape, wcs)``
tuple. The ``shape`` is simply the shape of the numpy array backing the map,
while ``wcs`` is an ``astropy.wcs.WCS`` (World Coordinate System) object that
attaches coordinates to the last two axes of the map
according to the FITS standard. This is defined in `Representations of world
coordinates in FITS <https://www.atnf.csiro.au/computing/software/wcs/WCS/wcs.pdf>`_
and `Representations of celestial coordinates in FITS <https://www.atnf.csiro.au/computing/software/wcs/WCS/ccs.pdf>`_.
Put simply, the world coordinate system describes how to translate from
pixel coordinates to sky ("world") coordinates.

Pixel Coordinates
-----------------

The pixels are represented by the last two axes of an
:py:class:`pixell.enmap.ndmap`.  We use row-major (C) ordering like ``numpy``,
so the second-to-last axis is the "y" (vertical) axis, while the last axis is
the "x" (horizontal) axis. A pixel's pixel coordinates are simply its index into
the last two axes, so the pixel ``map[10,20]`` would have pixel coordinates
``p=(10,20)``.

Pixel coordinates differ from the map index by changing continuosly
from pixel to pixel. Pixel centers have integer coordinates, while
the lower and upper edges of the pixel are 0.5 pixel coordinates
higher or lower respectively. So the pixel from the example above would
have its lower-left corner at ``(9.5,19.5)`` and its top-right corner
at ``(10.5,20.5)``. Hence, a map with ``shape = (ny,nx)`` covers a pixel
y coordinate range -0.5 to ny-0.5 and an x coordinate range of
-0.5 to nx-0.5.

A note on pixel and angle conventions
-------------------------------------

There is a mismatch between the pixel, angle and axis conventions
in ``enmap`` and the underlying FITS standard and ``astropy.wcs``.
``enmap`` uses zero-based pixels, row-major ordering and radians.
FITS uses one-based pixels, column-major ordering and degrees.
You will not normally need to worry about this, but it does matter
when manually dealing with ``wcs`` object internals.

In this documentation it will sometimes be useful to work with the FITS
convention. Variables in this convention will be uppercase, unless otherwise
stated.

Intermediate Coordinates
------------------------

Intermediate coordinates are equivalent to pixel coordinates, just
scaled to angle units. In most cases they are simply given by
subtracting ``crpix`` from the pixel coordinates, scaling them by ``cdelt``,
and adding ``crval`` to the result:::

  q = (p-crpix)*cdelt + crval

or if written out explicitly for the components,::

  q[i] = (p[i] - crpix[i])*cdelt[i] + crval[i]

where ``i`` runs over the values 0 (y) and 1 (x). In the FITS
convention this is::

  Q = (P-CRPIX)*CDELT + CRVAL

where ``P = p[::-1]+1``, ``CRPIX = crpix[::-1]+1``, ``CDELT = cdelt[::-1]*180/pi``,
``CRVAL = crval[::-1]*180/pi`` and ``Q = q[::-1]*180/pi``.

In either case, the result is that pixel coordinates are linearly mapped
to a flat, equirectangular coordinate system.

Sky Coordinates
---------------

The sky is curved while a map is flat, so one needs a projection to map
between them, and this is handled in the translation from intermediate
to sky coordinates. This is mainly controlled with the ``ctype``
parameter, which typically takes the form::

  CTYPE = ["RA---XXX", "DEC--XXX"]

where "XXX" describes the projection type. Common values are

* CAR: The Equirectangular projection. If ``crval[0]=0`` (so the
  reference point is on the equator), then this is equivalent to
  the Plate Carrée projection, and sky coordinates end up being
  the same as intermediate coordinates.
* CEA: Cylindrical Equal-Area projection. Like CAR, but pixels get
  taller as one approaches the poles to keep their physical area
  constant.
* TAN: Tangent plane (gnominic) projection. Only covers at most
  half the sky.  This is the projection a pinhole camera produces.

and many others. See `Representations of celestial coordinates in FITS <https://www.atnf.csiro.au/computing/software/wcs/WCS/ccs.pdf>`_
for much more details.

Note on the reference point
---------------------------

The point given by ``crval`` is called the "reference point", and has
several roles.

1. It defines the intermediate coordinates together with ``crpix``
   and ``cdelt``.
2. It defines the center point of the projection. That doesn't
   mean that it will be the center of the actual map, but it's
   the center of the full logical projection the map is a window
   into. For example, in the orthographic projection (SIN), which
   corresponds to looking at a sphere from an infinite distance,
   the reference point specifies which part of the sphere is
   facing the viewer. Similarly, a CAR projection corresponds to
   the sky rotated until the reference point faces the viewer,
   and then unrolled along a cylinder. This means that if the
   reference point is not on the equator (dec=0°), then the
   equator will not be a horizontal line in the resulting map.
   In almost all choices one will want to put the reference point
   on the equator in practice.
3. It defines the area of validity for the coordinates. Areas
   more than 180° away from the reference point are in theory
   invalid and may cause problems (e.g. with drawing the coordinate
   grid in ``enplot``), but are supported as an extension for
   cylindrical coordinates in many ``enmap`` functions like
   :py:meth:`pixell.enmap.pix2sky` and :py:meth:`pixell.enmap.sky2pix`.

Supported geometries
--------------------

There are three tiers of geometry support in ``pixell``:

* Bronze: Most functions work, but some are slower or use
  more memory, and a few give invalid results. For example,
  some functions assume that every pixel has valid coordinats,
  but this is not always the case. Example: MOL (Mollweide)
  or ARC (zenithal).
* Silver: Everything works, except only slow, low-accuracy
  versions of ``map2alm`` spherical harmonics analysis is
  available. Spherical harmonics anlaysis requires integration
  weights to be avilable in ``ducc``, and this is currently
  only the case for a few pixelizations. Example: CEA,
  most variants of CAR.
* Gold: Full support. This is only available for specific
  variants of the Equirectangular CAR projection. In particular,
  these things must be satisfied:

   1. The projection must be CAR
   2. There must be a whole number of pixels around the sky in the
      RA direction, so 360/CDELT[0] (or equivalently pi/cdelt[1])
      must be an integer
   3. There must be either a pixel edge or a pixel center at the
      poles. These correspond to different integration weights.
      An edge at both poles corresponds to Fejer's first rule.
      A center at both poles corresponds to Clenshaw-Curtis.
      We recommend the former, since it generalizes better when
      downsampling or upsampling a map. We call this variant
      of CAR "Fejer1".

To summarize, ``pixell`` works with a wide range of geometries,
but for full spherical harmonics support, we recommend CAR
maps in the Fejer1 pixelization.

Using geometries
----------------

The job of an ``enmap`` geometry is to allow translation between
pixel coordinates and sky coordinates. The most basic functions
that do this are ``pos = enmap.pix2sky(shape, wcs, pix)`` and
``pix = enmap.sky2pix(shape, wcs, pos)``. Here ``pix``
should be ``[{y,x},...]``, meaning it should be at
least 1-dimensional, where the first axis has length two and
contains the y and x pixel coordinates in that order. Similarly,
``pos`` should be ``[{dec,ra},...]```.

:py:meth:`pixell.enmap.sky2pix` tries
to ensure that no angle wrapping happens in the output, so that
there won't be a sudden 2π jump between the coordinates of
neighboring pixels. This has a small overhead, and sometimes
results in coordinates that are overall offset by some multiple
of 2π from what one might want, so this can be disabled by passing
``safe=False`` as an argument.

There are many higher-level functions built from these. The most
useful are

* :py:meth:`pixell.enmap.posmap`: returns a new enmap with shape
  ``[{dec,ra},ny,nx]``, containing the coordinates of each pixel.
* :py:meth:`pixell.enmap.pixsizemap`: returns the area of each pixel,
  in steradians.
* :py:meth:`pixell.enmap.distance_from`: returns the distance
  of each pixel from the closest of the given list of positions,
  and optionally the index of which point was closest.
* :py:meth:`pixell.enmap.corners` (alias ``enmap.box``): returns the
  coordinates of the bottom left and top right corners of the map.
  For cylindrical projections, this is the map's bounding box.
* :py:meth:`pixell.enmap.area`: returns the area of the map,
  in steradians.

The map geometry also enters into a large number of functions for
working with full ``enmap.ndmap`` objects. These are covered
`HERE <fixme>`.

Building geometries
-------------------

There are currently several ways of building geometries, which a
good deal of overlap between what they do. This will hopefully be
cleaned up in the future.

Explicit construction
^^^^^^^^^^^^^^^^^^^^^

You can construct wcs objects manually using :py:func:`pixell.wcsutils.explicit`,
which takes lower-case FITS ``wcs`` parameters as arguments, in the FITS convention.
For example, this constructs a full-sky CAR map with 0.5 arcmin Fejer1 pixelization.

.. code-block:: python
  
  shape = (180 * 120, 360 * 120)
  wcs   = wcsutils.explicit(
    ctype=["RA---CAR", "DEC--CAR"],
    crval=[0, 0],
    cdelt=[-0.5/60, 0.5/60],
    crpix=[180*120+1, 90*120+0.5]
  )

Notice how the arguments are in RA-dec ordering, in degrees, and with crpix counting
from 1, unlike the normal ``enmap`` functions.

geometry2
^^^^^^^^^

:py:func:`pixell.enmap.geometry2` makes it easy to construct geometries that
fulfill boundary conditions like Fejer1. It works by first constructing a
full-sky geometry, and then optionally cropping out a subset of interest from
that. For example, this constructs a full-sky CAR map with a 0.5 arcmin Fejer1
pixelization.::

  shape, wcs = enmap.geometry2(res=0.5*utils.arcmin)

This constructs a geometry covering -4°<dec<5°, 120°>RA>100° that is compatible
with the full-sky geometry above. By "compatible", we mean that they can be
cropped or padded to align perfectly with each other, with no interpolation needed.
This happens when every pixel in one geometry has integer pixel coordinates in the
other.::

  box = np.array([[-4,120],[5,100]])*utils.degree
  shape, wcs = enmap.geometry2(pos=box, res=0.5*utils.arcmin)

Notice that ``box`` has shape ``[{bottom-left,top-right},{dec,ra}]``,
and since the standard is for RA to increase towards the *left* in the
map, the RA bounds are given in descending order here. If box instead
has been given as ``np.array([[-4,100],[5,120]])*utils.degree``, then
the map would cover the same area, but RA would be increasing towards
the right instead, which usually isn't what you want.

You can also build geometriees by giving a center point, resolution and
shape. For example, this builds a small tangent plane (Gnomonic)
patch centered on RA=dec=0°.::

  shape, wcs = enmap.geometry2(pos=[0,0], res=0.5*utils.arcmin, shape=(101,101), proj="tan")

In this case we end up with the central pixel with coordinates of exactly
RA=dec=0°, because the TAN projection doesn't have any special boundary
conditions like Fejer1 to fulfil. If you tried the same thing with
``proj="car"``, then the central pixel could be up to a quarter pixel
away from the requested position to fulfil the boundary condition. This
can be avoided with the ``variant="any"`` argument. See the function's
full documentation for details.

thumbnail_geometry
^^^^^^^^^^^^^^^^^^

Convenience function for making small thumbnails appropriate for e.g.
object stacking. For example, this creates a 10 arcmin radius, 0.25 arcmin
resolution tangent-plane projection geometry centered on RA=dec=0°.::

  shape, wcs = enmap.thumbnail_geometry(r=10*utils.arcmin, res=0.25*utils.arcmin)

geometry
^^^^^^^^

This is ``geometry2``'s predecessor. It has a similar interface, but
is based around a reference point instead of a boundary condition.
It gives that reference point, which is by default at RA=dec=0°,
integer pixel coordinates. This ensures that different geometries
with the same projection and resolution are pixel-compatible, even
if the cover different parts of the sky, but it does not ensure that
they follow the north and south pole boundary conditions needed for
spherical harmonic anlaysis (``map2alm``). It's this limitation that
led to the creation of ``geometry2``.

You should avoid ``geometry``, and is relatives ``fullsky_geometry``
and ``band_geometry`` in favor of ``geometry2``. ``geometry`` may
be deprecated in the future.

Geometry manipulation
---------------------

When you slice, submap, downgrade, upgrade, etc. a map, the attached
geometry will be automatically updated to reflect this, but sometimes
it's useful to be able to manipulate geometries directly, without
having to construct a full map first. This is supported via the
following functions:

* ``enmap.downgrade_geometry(shape, wcs, n)``: Produce the same
  geometry you would get by using ``enmap.downgrade`` on the corresponding
  map. This is a geometry that covers the same area, but with ``n``
  times as low resolution.
* ``enmap.upgrade_geometry(shape, wcs, n)``: The inverse of
  ``downgrade_geometry``.
* ``enmap.subgeo(shape, wcs, box=box) or enmap.subgeo(shape, wcs, pixbox=pixbox)``:
  return the sub-geometry corresponding to the given rectangle,
  specified either as ``box`` (``[{bottom-left,top-right},{dec,ra}]``)
  or ``pixbox`` (``[{bottom-left,top-right},{y,x}]``).
* ``enmap.union_geometry(geometries)``: return the first geometry padded to
  contain all the others.

Geometry objects
----------------

The class ``enmap.Geometry`` encapsulates a ``(shape, wcs)`` pair, and
provides some of the ``enmap.ndmap`` interface. The purpose is to make
working with a geometry as similar to working with a map as possible,
with support for e.g. slicing. So far only a few methods have been
implemented, but this may improve in the future. Example usage:::

  geo = enmap.Geometry(shape, wcs)
  geo = geo[0:100:2,100,200:2] # crop and downgrade
  shape, wcs = geo

Geometry I/O
------------

Geometries can be read and written to disk much like maps. They
are represented as a FITS header without the corresponding FITS
body, so they are tiny and fast to read.::

* ``enmap.write_map_geometry(fname, shape, wcs)``
* ``shape, wcs = enmap.read_map_geometry(fname)``

Useful geometry concepts
------------------------

Ring-compatible
^^^^^^^^^^^^^^^

``ducc``'s spherical harmonics transforms (both ``alm2map`` and ``map2alm``)
require the map to consist of rings of constant-declination pixels with
constant pixel spacing inside each ring, and an integer number of pixels
around the sky in the RA direction (though the number of pixels per ring
can vary from ring to ring).

Cylindrical projections have this property, e.g. CAR, CEA, MER. Most
pseudocylindrical projections also satisfy this property, as long as
the invalid pixels that are inside the bounding rectangle but not actually
part of the projection aren't included in the rings. A popular example
of a projection like this is Mollweide (MOL). SHTs on these aren't directly
supported by ``pixell`` currently. Expensive repixelization is necessary.

Quadrature-compatible
^^^^^^^^^^^^^^^^^^^^^

While spherical harmonic synthesis (``alm2map``) operations are simply a
sum over multipoles per pixel, spherical harmonic analysis (``map2alm``)
is an *integral* over the sky, and therefore requires quadrature weights
to be accurate. To first order, these are just the area of each pixel,
but this approximation isn't very good. ``ducc`` provides optimal quadrature
weights for a limited set of ring-compatible pixelizations, including
HEALPix (not supported by ``enmap.ndmaps``) and a few variants of CAR
that have the following properties:

* Ring-compatible
* Either a pixel center or a pixel edge at the north and south poles.

There are 4 possible combinations of these boundary conditions.

* Edge and edge = Fejer's first quadrature rule = Fejer1 = F1
* Center and center = Clenshaw-Curtis = CC
* Edge and center = McEwen and Wiaux = MW
* Center and edge = MWflip

(A few exotic variants like Fejer's second rule and Driscoss-Healy
are also supported, but unlikely to be useful).

``enmap.geometry2`` produces Fejer1 geometries by default.

Downgradable
^^^^^^^^^^^^

We call a geometry "downgradable" when it preserves its nice properties
when downgraded by small integer factors like 2 or 4. Of the
quadature-compatible geometries, only Fejer1 has this property.
By "downgrade", we mean the action of replacing each group of n×n pixels
with a single pixel with the average of their value. Below is an example
of what happens to a 1D Fejer1 geometry when downgraded by a factor of 2.::

       10.0   -20.0    7.5    12.1
    |---*---|---*---|---*---|---*---|
  -90     -45       0      45      90

                    ↓

           -5.0            9.8
    |-------*-------|-------*-------|
  -90               0              90

Each interval represents one pixel. The pixel edges are indicated with ``|``
and their centers with ``*``. The coordinates of the pixel edges
given below, and example pixel values above. As you can see, we started with
a pixel edge at the north and south pole (-90° and +90°), and this was still
the case after downgrading. This would not be the case with e.g.
Clenshaw-Curtis:::

       10.0   -20.0    7.5    12.1
    |...*---|---*---|---*---|---*...|
  -120    -60       0      60      120

                    ↓

           -5.0            9.8
    |...----*-------|-------*----...|
  -120              0              120

Here the north and south pole now fall in the center of the first and last
pixel before downgrading, but are 3/4 the way to the edge after downgrading.
Hence, while a downgraded Fejer1 is still Fejer1, a downgraded CC is never CC.

(Note: Aside from these concerns, a geometry is only downgradable if the downgrade
factor is a factor of both the y and x size of the map.)

Pixel-compatible
^^^^^^^^^^^^^^^^

We say that two geometries are pixel-compatible if they are windows into the same
underlying fullsky geometry, possibly with a cyclic shift in RA for cylindrical
projections. It's convenient if most of the maps one works with are pixel-compatible,
since it lets one easily move map values from one to the other with just copying,
no interpolation needed. ``wcsutils.is_compatible(wcs1, wcs2)`` returns True if
two geometries are compatible. One can cheaply and project a map onto a compatible
geometry with ``enmap.extract``. For non-compatible geometries, the heavier,
interpolation-based ``enmap.project`` must be used.

Separable
^^^^^^^^^

A geometry is "separable" if dec is only a function of y, and RA is only a function
of x. Some operations on separable maps are much faster and use less memory because
only ``ny+nx`` values need to be calculated, istead of ``2*ny*nx`` values. These
functions often have arguments like ``broadcastable`` to allow working with the
smaller representation as long as possible. For example, if ``broadcastable=True``
is passed to ``enmap.pixsizemap``, then it will return a result with shape
``(ny,1)`` if the geometry is separable, instead of the normal ``(ny,nx)``.
Since ``(ny,1)`` broadcasts to ``(ny,nx)``, most code can work with this can work
can benefit from cheaper operations on the former without needing to be modified.
