.. _UsagePage:

=====
Usage
=====

.. sectnum:: :start: 1




Any map can be completely specified by two objects, a familiar numpy array (of at least two dimensions) whose two trailing dimensions correspond to two coordinate axes of the map, and a ``wcs`` object that specifies the World Coordinate System. The latter specifies the correspondence between pixels and physical sky coordinates. This library allows for the manipulation of an object ``ndmap``, which has all the properties of numpy arrays but is in addition enriched by the ``wcs`` object. The ``shape`` of the numpy array and the ``wcs`` completely specifies the geometry and footprint of a map of the sky.

I've listed below common operations that would be useful to demonstrate here. TODO: Finish this! (See :ref:`ReferencePage` for a dump of all member functions)

Reading maps from disk
--------

Sky coordinate and pixel conversions
--------

Distance from center -- ``modrmap``
~~~~~~

Fourier modes of the map
--------

Absolute wave-number -- ``modlmap``
~~~~~~

Filtering maps
--------

Manipulating map geometries
----------

Resampling maps
--------

Masking and windowing
--------

Flat-sky diagnostic power spectra
---------

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



