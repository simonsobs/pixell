.. _UsagePage:

=====
Usage
=====

.. sectnum:: :start: 1

Any map can be completely specified by two objects, a numpy array (of at least two dimensions) whose two trailing dimensions correspond to two coordinate axes of the map, and a ``wcs`` object that specifies the World Coordinate System. The latter specifies the correspondence between pixels and physical sky coordinates. This library allows for the manipulation of an object ``ndmap``, which has all the properties of numpy arrays but is in addition enriched by the ``wcs`` object (specifically an instantiation of Astropy's ``astropy.wcs.wcs.WCS`` object). The ``shape`` of the numpy array and the ``wcs`` completely specifies the geometry and footprint of a map of the sky.

All ``ndmap`` s must have at least two dimensions. The trailing two axes are interpreted as the Y (typically, declination) and X (typically, right ascension) axes. Maps can have arbitrary number of leading dimensions, but many of ``pixell``' CMB-related tools interpret a 3D array of shape ``(ncomp,Ny,Nx)`` to consist of three ``Ny`` x ``Nx`` maps of intensity, polarization Q and U Stokes parameters in that order.

TODO: I've listed below common operations that would be useful to demonstrate here.  Finish this! (See :ref:`ReferencePage` for a dump of all member functions)

Maps are extensions of ``numpy`` arrays
---------------------------------------

Apart from all the numpy functionality, ``ndmap`` comes with a host of
additional attributes and functions that utilize the information in the
WCS. This usage guide will demonstrate how such maps can be manipulated using
``pixell``. While reading about all this additional functionality, please keep
in mind that the great thing about ``ndmap`` s is that they can be used like
regular numpy arrays. For example if you slice the trailing axes of the array
like you would a numpy array (e.g. ``imap[:100,:100]``), you are effectively slicing out a section of the
map to produce a new ``ndmap`` with a reduced footprint.

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
		>>> omap = enmap.enmap(omap,wcs) # restore the wcs
		>>> omap = enmap.samewcs(omap,imap) # another way to restore the wcs



Reading maps from disk
----------------------

An entire map in ``FITS`` or ``HDF`` format can be loaded using ``read_map``, which is found in the module ``pixell.enmap``. The ``enmap`` module contains the majority of map manipulation functions.

.. code-block:: python

		from pixell import enmap
		imap = enmap.read_map("map_on_disk.fits")

Alternatively, one can select a rectangular region specified through its bounds using the ``box`` argument,

.. code-block:: python

		import numpy as np
		dec_min = -5 ; ra_min = -5 ; dec_max = 5 ; ra_max = 5
		# All coordinates in pixell are specified in radians
		box = np.deg2rad([[dec_min,ra_min],[dec_max,ra_max])) 
		imap = enmap.read_map("map_on_disk.fits",box=box) 


Note the convention used to define coordinate boxes in pixell. To learn how to use a pixel box or a numpy slice, please read the docstring for ``read_map``.

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
~~~~~~

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
~~~~~

Using the ``enmap.posmap`` function, you can get a map of shape (2,Ny,Nx)
containing the coordinate positions in radians of each pixel of the map.

.. code-block:: python

		>>> posmap = imap.posmap()
		>>> dec = posmap[0] # declination in radians
		>>> ra = posmap[1] # right ascension in radians


Pixel map
~~~~~

Using the ``enmap.pixmap`` function, you can get a map of shape (2,Ny,Nx)
containing the integer pixel coordinates of each pixel of the map.

.. code-block:: python

		>>> pixmap = imap.pixmap()
		>>> pixy = posmap[0] 
		>>> pixx = posmap[1] 


Distance from center -- ``modrmap``
~~~~~~

Using the ``enmap.modrmap`` function, you can get a map of shape (Ny,Nx)
containing the physical coordinate distance of each pixel from a given reference
point specified in radians. If the reference point is unspecified, the distance
of each pixel from the center of the map is returned.

.. code-block:: python

		>>> modrmap = imap.modrmap() # 2D map of distances from center


Fourier operations
--------

Maps can be 2D Fourier-transformed for manipulation in Fourier space. The 2DFT
of the (real) map is generally a complex ``ndmap`` with the same shape as the
original map (unless a real transform function is used). To facilitate 2DFTs, there are functions that do the Fourier transforms themselves,
and functions that provide metadata associated with such transforms.

What are the wavenumbers or multipoles of the map?
~~~~~~

Since an `ndmap` contains information about the physical extent of the map and
the physical width of the pixels, the discrete frequencies corresponding to its
numpy array need to be converted to physical wavenumbers of the map.

This is done by the ``laxes`` function, which returns the wavenumbers
along the Y and X directions. The ``lmap`` function returns a map of all the
``(ly,lx)`` wavenumbers in each pixel of the Fourier-space map. The ``modlmap``
function returns the "modulus of lmap", i.e. a map of the distances of each
Fourier-pixel from ``(ly=0,lx=0)``.

FFTs and inverse FFTs
~~~~~~~~~

You can perform a fast Fourier transform of an (...,Ny,Nx) dimensional `ndmap`
to return an (...,Ny,Nx) dimensional complex map using ``enmap.fft`` and
``enmap.ifft`` (inverse FFT).

Filtering maps in Fourier space
--------

A filter can be applied to a map in three steps:

1. prepare a Fourier space filter ``kfilter``
2. Fourier transform the map ``imap`` to ``kmap``
3. multiply the filter and k-map
4. inverse Fourier transform the result

Manipulating map geometries
----------

Resampling maps
--------

Masking and windowing
--------

Flat-sky diagnostic power spectra
---------

Curved-sky operations
--------

Spherical harmonic transforms
~~~~~~~~

Filtering in spherical harmonic space
~~~~~~~~

The resulting spherical harmonic `alm` coefficients of an SHT are stored in the
same convention as with ``HEALPIX``, so one can use ``healpy.almxfl`` to apply
an isotropic filter to an SHT.

Diagnostic power spectra
~~~~~~~~


Reprojecting maps
---------

Map re-centering
~~~~~~

Postage stamp extraction
~~~~~~

To and from ``healpix``
~~~~~~

Simulating maps
----------

Gaussian random field generation
~~~~~

Lensing and delensing
~~~~~

Point source simulation
~~~~~



