.. _UsagePage:

=====
Usage
=====

.. sectnum:: :start: 1

The ``ndmap`` object
--------------------

The ``pixell`` library supports manipulation of sky maps that are
represented as 2-dimensional grids of rectangular pixels.  The
supported projection and pixelization schemes are a subset of the
schemes supported by FITS conventions. In addition, we provide
support for a `plain' coordinate system, corresponding to a
Cartesian plane with identically shaped pixels (useful for true
flat-sky calculations).

In ``pixell``, a map is encapsulated in an ``ndmap``, which combines
two objects: a numpy array (of at least two dimensions) whose two
trailing dimensions correspond to two coordinate axes of the map, and
a ``wcs`` object that specifies the World Coordinate System.  The
``wcs`` component is an instance of Astropy's ``astropy.wcs.wcs.WCS``
class.  The combination of the ``wcs`` and the ``shape`` of the numpy
array completely specifies the footprint of a map of the sky, and is
called the ``geometry``.  This library helps with manipulation of
``ndmap`` objects in ways that are aware of and preserve the validity
of the wcs information.

``ndmap`` as an extension of ``numpy.ndarray``
``````````````````````````````````````````````

The ``ndmap`` class extends the ``numpy.ndarray`` class, and thus has
all of the usual attributes (``.shape``, ``.dtype``, etc.) of an
``ndarray``.  It is likely that an ``ndmap`` object can be used in any
functions that usually operate on an ``ndarray``; this includes the
usual numpy array arithmetic, slicing, broadcasting, etc.

.. code-block:: python

   >>> from pixell import enmap
   >>> #... code that resulted in an ndmap called imap
   >>> print(imap.shape, imap.wcs)
   (100, 100) :{cdelt:[1,1],crval:[0,0],crpix:[0,0]}
   >>> imap_extract = imap[:50,:50]   # A view of one corner of the map.
   >>> imap_extract *= 1e6            # Re-calibrate. (Also affects imap!)

An ``ndmap`` must have at least two dimensions. The two right-most
axes represent celestial coordinates (typically Declination and Right
Ascension).  Maps can have arbitrary number of leading dimensions, but
many of the ``pixell`` CMB-related tools interpret 3D arrays with
shape ``(ncomp,Ny,Nx)`` as representing ``Ny`` x ``Nx`` maps of
intensity, polarization Q and U Stokes parameters, in that order.

Note that ``wcs`` information is correctly adjusted when the array is
sliced; for example the object returned by ``imap[:50,:50]`` is a view
into the ``imap`` data attached to a new ``wcs`` object that correctly
describes the footprint of the extracted pixels.

Apart from all the numpy functionality, ``ndmap`` comes with a host of
additional attributes and functions that utilize the WCS
information.

``ndmap.wcs``
`````````````

The ``wcs`` information describes the correspondence between celestial
coordinates (typically the Right Ascension and Declination in the
Equatorial system) and the pixel indices in the two right-most axes.
In some projections, such as CEA or CAR, rows (and columns) of the
pixel grid will often follow lines of constant Declination (and Right
Ascension).  In other projections, this will not be the case.

The WCS system is very flexible in how celestial coordinates may be
associated with the pixel array.  By observing certain conventions, we
can make life easier for users of our maps.  We recommend the
following:

- The first pixel, index [0,0], should be the one that you would
  normally display (on a monitor or printed figure) in the lower
  left-hand corner of the image.  The pixel indexed by [0,1] should
  appear to the right of [0,0], and pixel [1,0] should be above pixel
  [0,0].  (This recommendation originates in FITS standards
  documentation.)
- When working with large maps that are not near the celestial poles,
  Right Ascension should be roughly horizontal and Declination should
  be roughly vertical.  (It should go without saying that you should
  also present information "as it would appear on the sky", i.e. with
  Right Ascension increasing to the left!)

The examples in the rest of this document are designed to respect
these two conventions.

TODO: I've listed below common operations that would be useful to demonstrate here.  Finish this! (See :ref:`ReferencePage` for a dump of all member functions)

Creating an ``ndmap``
---------------------

To create an empty ``ndmap``, call the ``enmap.zeros`` or
``enmap.empty`` functions and specify the map shape as well as the
pixelization information (the WCS).  Here is a basic example:

.. code-block:: python

   >>> from pixell import enmap, utils
   >>> box = np.array([[-5,10],[5,-10]]) * utils.degree
   >>> shape,wcs = enmap.geometry(pos=box,res=0.5 * utils.arcmin,proj='car')
   >>> imap = enmap.zeros((3,) + shape, wcs=wcs)

In this example we are requesting a pixelization that spans from -5
to +5 in declination, and +10 to -10 in Right Ascension.  Note that we
need to specify the Right Ascension coordinates in decreasing order,
or the map, when we display it with pixel [0,0] in the lower left-hand
corner, will not have the usual astronomical orientation.

For more information on designing the geometry, see
:ref:`geometry-section`.

Passing maps through functions that act on ``numpy`` arrays
-----------------------------------------------------------

You can also perform
arithmetic with and use functions that act on numpy arrays. In most situations,
functions that usually act on numpy arrays will return an ``ndmap`` when an
``ndmap`` is passed to it in lieu of a numpy array. In those situations where
the WCS information is removed, one can always add it back like this:

.. code-block:: python

		>>> from pixell import enmap
		>>> #... code that resulted in an ndmap called imap
		>>> print(imap.shape, imap.wcs)
		(100, 100) :{cdelt:[1,1],crval:[0,0],crpix:[0,0]}
		>>> omap = some_function(imap)
		>>> print(omap.wcs)
		Traceback (most recent call last):
		AttributeError: 'numpy.ndarray' object has no attribute 'wcs'
		>>> # Uh oh, the WCS information was removed by some_function
		>>> omap = enmap.ndmap(omap, wcs) # restore the wcs
		>>> omap = enmap.samewcs(omap, imap) # another way to restore the wcs
		>>> # This does the same thing, but force-copies the data array.
		>>> omap = enmap.enmap(omap, wcs)

Note that ``ndmap`` and ``samewcs`` will not copy the underlying data
array if they don't have to; the returned object will reference the
same memory used by the input array (as though you had done
numpy.asarray).  In contrast, ``enmap.enmap`` will always create a
copy of the input data.


Reading maps from disk
----------------------

An entire map in ``FITS`` or ``HDF`` format can be loaded using ``read_map``, which is found in the module ``pixell.enmap``. The ``enmap`` module contains the majority of map manipulation functions.

.. code-block:: python

		>>> from pixell import enmap
		>>> imap = enmap.read_map("map_on_disk.fits")

Alternatively, one can select a rectangular region specified through its bounds using the ``box`` argument,

.. code-block:: python

		>>> import numpy as np
		>>> from pixell import utils
		>>> dec_min = -5 ; ra_min = -5 ; dec_max = 5 ; ra_max = 5
		>>> # All coordinates in pixell are specified in radians
		>>> box = np.array([[dec_min,ra_min],[dec_max,ra_max])) * utils.degree
		>>> imap = enmap.read_map("map_on_disk.fits",box=box) 


Note the convention used to define coordinate boxes in pixell. To learn how to
use a pixel coordinate box or a numpy slice, please read the docstring for ``read_map``.

Inspecting a map
----------------

An ``ndmap`` has all the attributes of a ``ndarray`` numpy array. In particular, you can inspect its shape.

.. code-block:: python

		>>> print(imap.shape)
		(3,500,1000)

Here, ``imap`` consists of three maps each with 500 pixels along the Y axis and 1000 pixels along the X axis. One can also inspect the WCS of the map,

.. code-block:: python

		>>> print(imap.wcs)
		car:{cdelt:[0.03333,0.03333],crval:[0,0],crpix:[500.5,250.5]}

Above, we learn that the map is represented in the ``CAR`` projection system and what the WCS attributes are.

Selecting regions of the sky
----------------------------

If you know the pixel coordinates of the sub-region you would like to select,
the cleanest thing to do is to slice it like a numpy array.

.. code-block:: python

		>>> imap = enmap.zeros((1000,1000))
		>>> print(imap.shape)
		(1000,1000)
		>>> omap = imap[100:200,50:80]
		>>> print(omap.shape)
		(100, 30)


However, if you only know the physical coordinate bounding box in radians, you
can use the ``submap`` function.

.. code-block:: python

		>>> box = np.array([[dec_min,ra_min],[dec_max,ra_max]]) # in radians
		>>> omap = imap.submap(box)
		>>> omap = enmap.submap(imap,box) # an alternative way


Relating pixels to the sky
--------------------------

The geometry specified through ``shape`` and ``wcs`` contains all the information to get properties of the map related to the sky. ``pixell`` always specifies the Y coordinate first. So a sky position is often in the form ``(dec,ra)`` where ``dec`` could be the declination and ``ra`` could be the right ascension in radians in the equatorial coordinate system.

Conversions
```````````

The pixel corresponding to ra=180,dec=20 can be obtained like

.. code-block:: python

		>>> dec = 20 ; ra = 180
		>>> coords = np.deg2rad(np.array((dec,ra)))
		>>> ypix,xpix = enmap.sky2pix(shape,wcs,coords)

Note that you don't need to pass each dec,ra separately. You can pass a large number of coordinates for a vectorized conversion. In this case `coords` should have the shape (2,Ncoords), where Ncoords is the number of coordinates you want to convert, with the first row containing declination and the second row containing right ascension. Also, the returned pixel coordinates are in general fractional.


Similarly, pixel coordinates can be converted to sky coordinates

.. code-block:: python

		>>> ypix = 100 ; xpix = 300
		>>> pixes = np.array((ypix,xpix))
		>>> dec,ra = enmap.pix2sky(shape,wcs,pixes)

with similar considerations as above for passing a large number of coordinates.



Position map
````````````

Using the ``enmap.posmap`` function, you can get a map of shape (2,Ny,Nx)
containing the coordinate positions in radians of each pixel of the map.

.. code-block:: python

		>>> posmap = imap.posmap()
		>>> dec = posmap[0] # declination in radians
		>>> ra = posmap[1] # right ascension in radians


Pixel map
`````````

Using the ``enmap.pixmap`` function, you can get a map of shape (2,Ny,Nx)
containing the integer pixel coordinates of each pixel of the map.

.. code-block:: python

		>>> pixmap = imap.pixmap()
		>>> pixy = pixmap[0] 
		>>> pixx = pixmap[1] 


Distance from center -- ``modrmap``
```````````````````````````````````

Using the ``enmap.modrmap`` function, you can get a map of shape (Ny,Nx)
containing the physical coordinate distance of each pixel from a given reference
point specified in radians. If the reference point is unspecified, the distance
of each pixel from the center of the map is returned.

.. code-block:: python

		>>> modrmap = imap.modrmap() # 2D map of distances from center


Fourier operations
-------------------

Maps can be 2D Fourier-transformed for manipulation in Fourier space. The 2DFT
of the (real) map is generally a complex ``ndmap`` with the same shape as the
original map (unless a real transform function is used). To facilitate 2DFTs, 
there are functions that do the Fourier transforms themselves,
and functions that provide metadata associated with such transforms.

What are the wavenumbers or multipoles of the map?
``````````````````````````````````````````````````

Since an `ndmap` contains information about the physical extent of the map and
the physical width of the pixels, the discrete frequencies corresponding to its
numpy array need to be converted to physical wavenumbers of the map.

This is done by the ``laxes`` function, which returns the wavenumbers
along the Y and X directions. The ``lmap`` function returns a map of all the
``(ly,lx)`` wavenumbers in each pixel of the Fourier-space map. The ``modlmap``
function returns the "modulus of lmap", i.e. a map of the distances of each
Fourier-pixel from ``(ly=0,lx=0)``.

FFTs and inverse FFTs
`````````````````````

You can perform a fast Fourier transform of an (...,Ny,Nx) dimensional `ndmap`
to return an (...,Ny,Nx) dimensional complex map using ``enmap.fft`` and
``enmap.ifft`` (inverse FFT).

Filtering maps in Fourier space
-------------------------------

A filter can be applied to a map in three steps:

1. prepare a Fourier space filter ``kfilter``
2. Fourier transform the map ``imap`` to ``kmap``
3. multiply the filter and k-map
4. inverse Fourier transform the result


.. _geometry-section:

Building a map geometry
-----------------------

Patches
```````

You can create a geometry if you know what its bounding box and pixel size are:

.. code-block:: python

		>>> from pixell import enmap, utils
		>>> box = np.array([[-5,10],[5,-10]]) * utils.degree
		>>> shape,wcs = enmap.geometry(pos=box,res=0.5 * utils.arcmin,proj='car')

This creates a CAR geometry centered on RA=0d,DEC=0d with a width of
20 degrees, a height of 10 degrees, and a pixel size of 0.5
arcminutes.

Full sky
````````

You can create a full-sky geometry by just specifying the resolution:

.. code-block:: python

		>>> from pixell import enmap, utils
		>>> shape,wcs = enmap.fullsky_geometry(res=0.5 * utils.arcmin,proj='car')

This creates a CAR geometry with pixel size of 0.5 arcminutes that wraps around
the whole sky.

Declination-cut sky
```````````````````

You can create a geometry that wraps around the full sky but does not extend
everywhere in declination:

.. code-block:: python

		>>> shape,wcs = enmap.band_geometry(dec_cut=20*utils.degree, res=0.5 * utils.arcmin,proj='car')

This creates a CAR geometry with pixel size of 0.5 arcminutes that wraps around
the whole sky but is limited to DEC=-20d to 20d. The following creates the same
except with a declination extent from -60d to 30d.

.. code-block:: python

		>>> shape,wcs = enmap.band_geometry(dec_cut=np.array([-60,30])*utils.degree, res=0.5 * utils.arcmin,proj='car')


Resampling maps
---------------

:py:func:`pixell.enmap.resample` changes the number of pixels while keeping the
same footprint on the sky. The default FFT-based method preserves the power
spectrum up to the Nyquist frequency of the output:

.. code-block:: python

    >>> m = enmap.read_map("my_map.fits")
    >>> new_shape = (m.shape[-2] // 2, m.shape[-1] // 2)
    >>> m_half = m.resample(new_shape)   # ~half the pixels per axis

For coarser (degraded) resolution where you want to average pixels rather than
interpolate, use :py:func:`pixell.enmap.downgrade` instead:

.. code-block:: python

    >>> m_low = m.downgrade(2)   # average 2x2 blocks

See :doc:`objects <./objects>` for more on ``downgrade`` and ``upgrade``.

Masking and windowing
---------------------

A mask is an ``ndmap`` of boolean or float values indicating valid sky pixels.
Applying an apodization window smoothly tapers the map to zero at mask edges,
preventing Fourier-space ringing.

.. code-block:: python

    >>> from pixell import enmap, utils
    >>> m    = enmap.read_map("my_map.fits")
    >>> ivar = enmap.read_map("my_ivar.fits")

    >>> # Build a mask from the coverage map
    >>> mask = (ivar > 0.1 * np.max(ivar))

    >>> # Smooth the mask edges over 1 degree, and taper the boundary
    >>> win = enmap.apod_mask(mask, width=1.0 * utils.degree)
    >>> win = enmap.apod(win, width=5 * utils.arcmin)

    >>> # Apply window before computing power spectra
    >>> m_windowed = m * win
    >>> w2 = np.mean(win**2)   # normalization correction

See :doc:`masking <./masking>` for a full discussion including distance
transforms and mask growing/shrinking.

Flat-sky diagnostic power spectra
-----------------------------------

For a quick estimate of the power spectrum of a map patch, compute the 2D FFT
and bin azimuthally in :math:`|\ell|`:

.. code-block:: python

    >>> from pixell import enmap, utils
    >>> import numpy as np

    >>> m     = enmap.read_map("my_patch.fits")
    >>> m_ap  = enmap.apod(m, width=5 * utils.arcmin)   # apodize first

    >>> fmap  = enmap.fft(m_ap, normalize="phys")
    >>> p2d   = np.abs(fmap)**2
    >>> ls, cl = enmap.lbin(p2d, bsize=100)

See :doc:`fourier <./fourier>` for a full treatment of flat-sky Fourier analysis.

Curved-sky operations
---------------------

Spherical harmonic transforms
`````````````````````````````

For large patches or full-sky maps, use :py:mod:`pixell.curvedsky` to decompose
a map into spherical harmonic coefficients (alms):

.. code-block:: python

    >>> from pixell import enmap, curvedsky, utils
    >>> shape, wcs = enmap.geometry2(res=1 * utils.arcmin)   # full sky
    >>> m = enmap.read_map("my_fullsky_map.fits")

    >>> lmax = 5000
    >>> alm  = curvedsky.map2alm(m, lmax=lmax)

    >>> # Reconstruct the map from alms
    >>> m_rec = enmap.zeros(shape, wcs)
    >>> curvedsky.alm2map(alm, m_rec)

Filtering in spherical harmonic space
`````````````````````````````````````

The resulting spherical harmonic `alm` coefficients of an SHT are stored in the
same convention as with ``HEALPIX``, so one can use ``healpy.almxfl`` to apply
an isotropic filter to an SHT.

.. code-block:: python

    >>> from pixell import curvedsky, utils
    >>> import numpy as np

    >>> lmax  = 5000
    >>> ainfo = curvedsky.alm_info(lmax=lmax)
    >>> alm   = curvedsky.map2alm(m, lmax=lmax)

    >>> # Gaussian beam smoothing
    >>> fwhm  = 5 * utils.arcmin
    >>> sigma = fwhm / (8 * np.log(2))**0.5
    >>> ells  = np.arange(lmax + 1)
    >>> beam  = np.exp(-0.5 * ells * (ells + 1) * sigma**2)
    >>> alm_smooth = ainfo.lmul(alm, beam)

    >>> m_smooth = enmap.zeros(m.shape, m.wcs)
    >>> curvedsky.alm2map(alm_smooth, m_smooth)

See :doc:`harmonic <./harmonic>` for a full discussion of curved-sky analysis.

Diagnostic power spectra
````````````````````````

The angular power spectrum from alms:

.. code-block:: python

    >>> ainfo = curvedsky.alm_info(lmax=lmax)
    >>> cl    = ainfo.alm2cl(alm, alm)   # auto-spectrum: shape (lmax+1,)

Reprojecting maps
------------------

Map re-centering
````````````````

To extract a patch of the sky centered on a specific position, use
:py:meth:`pixell.enmap.ndmap.submap` with a coordinate box, or
:py:func:`pixell.reproject.thumbnails` for catalog-based cutouts with proper
tangent-plane reprojection:

.. code-block:: python

    >>> from pixell import enmap, utils
    >>> import numpy as np
    >>> m = enmap.read_map("my_map.fits")

    >>> # Extract a 2x2 degree patch centered on (dec=0, ra=0)
    >>> box = np.array([[-1, 1], [1, -1]]) * utils.degree
    >>> patch = m.submap(box)

Postage stamp extraction
````````````````````````

For catalog-based cutouts with reprojection onto a local tangent plane, use
:py:func:`pixell.reproject.thumbnails`:

.. code-block:: python

    >>> from pixell import reproject, utils
    >>> import numpy as np

    >>> # catalog: shape (N, 2), columns [dec, ra] in radians
    >>> catalog = np.array([[0.0, 0.0], [0.05, 0.1], [-0.03, 0.07]])

    >>> thumbs = reproject.thumbnails(
    ...     m, catalog, r=5*utils.arcmin, res=0.5*utils.arcmin
    ... )
    >>> print(thumbs.shape)  # (3, ny_thumb, nx_thumb)

See :doc:`thumbnails <./thumbnails>` for stacking and radial profiles.

To and from ``healpix``
```````````````````````
`pixell`` allows to reproject (convert) maps from CAR pixelization to HEALPix, and back.

Let's start by converting the ACT CAR map to HEALPix with `reproject.map2healpix`, 
optionally including a rotation from Celestial/Equatorial (native ACT) to Galactic 
(native Planck) coordinates.

We will assume that the ACT map is stored in a file called "act_d56_map.fits".	

.. code-block:: python

	>>> from pixell import reproject, enmap
	>>> import healpy as hp
	>>> act_car_map = enmap.read_map("act_d56_map.fits")
	>>> healpix_map = reproject.map2healpix(act_car_map, nside=2048, lmax=4000)

We can then plot the HEALPix map using healpy's mollview function to verify the result.:

.. code-block:: python

	>>> hp.mollview(healpix_map, title="ACT D56 Map in HEALPix", min = -300, max = 300)


Now we can reproject the HEALPix map back to CAR pixelization using `reproject.healpix2map`.
We can specify the shape, wcs, lmax as well as whether we want to include rotation. 
The method option controls how to interpolate between the input and output pixelization; 
the default option is harm, which uses spherical harmonics to do the interpolation.

In this example we also rotate from Galactic coordinates to Celestial with `rot="gal,cel"`.

.. code-block:: python

	>>> car_map_reprojected = reproject.healpix2map(healpix_map, shape=act_car_map.shape, wcs=act_car_map.wcs, lmax=4000, rot="gal,cel", method="harm")


In the case where the interpolation didn't work very well due to bright sources in the 
map, we can use a different method, like spline. Keep in mind that spline might not 
necessarily preserve power.

.. code-block:: python

	>>> car_map_reprojected_spline = reproject.healpix2map(healpix_map, shape=act_car_map.shape, wcs=act_car_map.wcs, lmax=4000, rot="gal,cel", method="spline")

Simulating maps
---------------

See :doc:`simulation <./simulation>` for a full treatment. Quick-start examples:

Gaussian random field generation
````````````````````````````````

.. code-block:: python

    >>> from pixell import enmap, curvedsky, powspec, utils
    >>> import numpy as np

    >>> shape, wcs = enmap.geometry2(res=1.5 * utils.arcmin)

    >>> # Read theory power spectrum (CAMB format)
    >>> ps  = powspec.read_spectrum("camb_lensedCls.dat", scale=True)

    >>> # Curved-sky CMB simulation
    >>> sim = curvedsky.rand_map(shape, wcs, ps[0:1, 0:1], lmax=6000, seed=1)

Lensing and delensing
`````````````````````

.. code-block:: python

    >>> from pixell import lensing, curvedsky, powspec, utils
    >>> import numpy as np

    >>> # Draw unlensed CMB and lensing potential alms
    >>> ps      = powspec.read_spectrum("camb_scalCls.dat", scale=True)
    >>> cmb_alm = curvedsky.rand_alm(ps[0:1, 0:1], lmax=6000, seed=1)
    >>> phi_alm = curvedsky.rand_alm(ps[0:1, 0:1], lmax=6000, seed=2)

    >>> # Apply curved-sky lensing
    >>> m_lensed = lensing.lens_map_curved(
    ...     shape, wcs, phi_alm, cmb_alm, lmax=6000
    ... )

Point source simulation
```````````````````````

.. code-block:: python

    >>> from pixell import enmap, pointsrcs, utils
    >>> import numpy as np

    >>> shape, wcs = enmap.geometry2(
    ...     pos=np.array([[-5, -5], [5, 5]]) * utils.degree,
    ...     res=0.5 * utils.arcmin,
    ... )

    >>> # 50 point sources at random positions
    >>> rng  = np.random.default_rng(0)
    >>> poss = rng.uniform(-5, 5, size=(2, 50)) * utils.degree  # (2, nsrc)
    >>> amps = rng.uniform(100, 500, size=(1, 50))              # (1, nsrc)

    >>> # Gaussian beam profile
    >>> fwhm    = 1.4 * utils.arcmin
    >>> sigma   = fwhm / (8 * np.log(2))**0.5
    >>> r       = np.linspace(0, 5 * sigma, 500)
    >>> profile = np.array([r, np.exp(-0.5 * r**2 / sigma**2)])

    >>> sim = pointsrcs.sim_objects(shape, wcs, poss, amps, profile)


