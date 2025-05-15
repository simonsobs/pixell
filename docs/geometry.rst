Sky Geometry
============

An `enmap.ndmap` represents one or more images of the sky, with a pixelization
described by a "geometry" represented as a `(shape, wcs)` tuple. The `shape` is
simply the shape of the numpy array backing the map, while `wcs` is an
`astropy.wcs.WCS` (World Coordinate System) object that attaches coordinates
to the last two axes of the map
according to the FITS standard. This is defined in [Representations of world
coordinates in FITS](https://www.atnf.csiro.au/computing/software/wcs/WCS/wcs.pdf)
and [Representations of celestial coordinates in FITS](https://www.atnf.csiro.au/computing/software/wcs/WCS/ccs.pdf). Put simply, the world coordinate system describes how to translate from
pixel coordinates to sky ("world") coordinates.

Pixel Coordinates
-----------------

The pixels are represented by the last two axes of an `enmap.ndmap`.
We use row-major (C) ordering like `numpy`, so the second-to-last axis
is the "y" (vertical) axis, while the last axis is the "x" (horizontal)
axis. A pixel's pixel coordinates are simply its index into the last two
axes, so the pixel `map[10,20]` would have pixel coordinates `(10,20)`.

Pixel coordinates differ from the map index by changing continuosly
from pixel to pixel. Pixel centers have integer coordinates, while
the lower and upper edges of the pixel are 0.5 pixel coordinates
higher or lower respectively. So the pixel from the example above would
have its lower-left corner at `(9.5,19.5)` and its top-right corner
at `(10.5,20.5)`. Hence, a map with `shape = (ny,nx)` covers a pixel
y coordinate range `-0.5 to ny-0.5` and an x coordinate range of
`-0.5 to nx-0.5`.

Intermediate Coordinates
------------------------

