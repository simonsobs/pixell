.. _UsagePage:

=====
Usage
=====

.. sectnum:: :start: 1




Any map can be completely specified by two objects, a numpy array (of at least two dimensions) whose two trailing dimensions correspond to two coordinate axes of the map, and a ``wcs`` object that specifies the World Coordinate System. The latter specifies the correspondence between pixels and physical sky coordinates. This library allows for the manipulation of an object ``ndmap``, which has all the properties of numpy arrays but is in addition enriched by the ``wcs`` object (specifically an instantiation of Astropy's ``astropy.wcs.wcs.WCS`` object). The ``shape`` of the numpy array and the ``wcs`` completely specifies the geometry and footprint of a map of the sky.

All ``ndmap`` s must have at least two dimensions. The trailing two axes are interpreted as the Y (typically, declination) and X (typically, right ascension) axes. Maps can have arbitrary number of leading dimensions, but many of ``pixell``' CMB-related tools interpret a 3D array of shape ``(ncomp,Ny,Nx)`` to consist of three ``Ny`` x ``Nx`` maps of intensity, polarization Q and U Stokes parameters in that order.

Apart from all the numpy functionality, ``ndmap`` comes with a host of additional attributes and functions that utilize the information in the WCS. This usage guide will demonstrate how such maps can be manipulated using ``pixell``. 


TODO: I've listed below common operations that would be useful to demonstrate here.  Finish this! (See :ref:`ReferencePage` for a dump of all member functions)

Reading maps from disk
--------

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
--------

An ``ndmap`` has all the attributes of a ``ndarray`` numpy array. In particular, you can inspect its shape.

.. code-block:: python

		> print(imap.shape)
		(3,500,1000)

Here, ``imap`` consists of three maps each with 500 pixels along the Y axis and 1000 pixels along the X axis. One can also inspect the WCS of the map,

.. code-block:: python

		> print(imap.wcs)
		car:{cdelt:[0.03333,0.03333],crval:[0,0],crpix:[500.5,250.5]}

Above, we learn that the map is represented in the ``CAR`` projection system and what the WCS attributes are.
   
Relating pixels to the sky
--------

The geometry specified through ``shape`` and ``wcs`` contains all the information to get properties of the map related to the sky. ``pixell`` always specifies the Y coordinate first. So a sky position is often in the form ``(dec,ra)`` where ``dec`` could be the declination and ``ra`` could be the right ascension in radians in the equatorial coordinate system.

Conversions
~~~~~~

The pixel corresponding to ra=180,dec=20 can be obtained like

.. code-block:: python

		> dec = 20 ; ra = 180
		> coords = np.deg2rad(np.array((dec,ra)))
		> ypix,xpix = enmap.sky2pix(shape,wcs,coords)

Note that you don't need to pass each dec,ra separately. You can pass a large number of coordinates for a vectorized conversion. In this case `coords` should have the shape (2,Ncoords), where Ncoords is the number of coordinates you want to convert, with the first row containing declination and the second row containing right ascension. Also, the returned pixel coordinates are in general fractional.

Similarly, pixel coordinates can be converted to sky coordinates

.. code-block:: python

		> ypix = 100 ; xpix = 300
		> pixes = np.array((ypix,xpix))
		> dec,ra = enmap.pix2sky(shape,wcs,pixes)

with similar considerations as above for passing a large number of coordinates.



Position map
~~~~~



Pixel map
~~~~~

Distance from center -- ``modrmap``
~~~~~~

Fourier modes of the map
--------

Absolute wave-number -- ``modlmap``
~~~~~~

Filtering maps
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

Filtering
~~~~~~~~

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



