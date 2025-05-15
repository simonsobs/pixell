Loading and working with maps
=============================

:py:mod:`pixell` is a library designed around the manipulation, creation, and analysis of map
products, primarily those from CMB surveys like ACT and SO. Pixell can load maps in three
major formats: fits, HDF5, and serialized numpy arrays, though the latter is generally
discouraged.

To get started on loading a map, you may wish to download a sample map. There are many
maps available from the ACT DR5 and DR6 data releases, we recommend
`this one <https://lambda.gsfc.nasa.gov/data/act/maps/published/act_dr6.02_std_AA_day_pa4_f150_4way_coadd_map.fits>`_
for testing purposes.

Reading Maps
------------

To read the map from disk into a pixell :py:class:`pixell.enmap.ndmap`:

.. code-block:: python

    from pixell import enmap

    act_dr6 = enmap.read_map("act_dr6.02_std_AA_day_pa4_f150_4way_coadd_map.fits")

It is important to recognise that the :py:func:`pixell.enmap.read_map` will only load the first
HDU of a fits map. To read different HDUs into separate
:py:class:`pixell.enmap.ndmap` objects, you can use the
:py:func:`pixell.enmap.read_fits` function directly.

This :py:func:`pixell.enmap.read_map` has a number of important (optional) parameters:

- `delayed=False`, a boolean stating whether you wish to delay reading the data
  in the map or read it all at once into memory (data is only read when required
  if that is the case; this can be slower if you are making lots of small map
  accesses).
- `box=None`, whether to only read out a sub-map based upon a particular geometry.
- `sel=None`, a mask array if you only want to read certain elements of the map. Useful
  if you wish to slice a map to only read out one Stokes component, for example.
- `geometry=None`, whether to force a particlar geometry onto your map; generally
  this is not used.

There is much more information on the :py:class:`pixell.enmap.ndmap` class in the
:doc:`object documentation <./objects>`, but for now you can think of it as a coupled
numpy array - storing the underlying map data (e.g. the flux in each pixel) - and an
astropy WCS object - storing the sky geometry those pixels are stored within. There is
:doc:`further documentation on geometry <./geometry>`, too.

Creating Empty Maps
-------------------

As well as reading maps, you can always create empty maps, should you wish to fill
them with your own data. Similar to numpy arrays, we provide utilities to create
maps:

.. code-block:: python

    from pixell import enmap

    empty = enmap.empty((32, 32))
    zeros = enmap.zeros((128, 64))
    ones = enmap.ones((128, 64))

You can specify both a `wcs` and a `dtype` parameter here, to set the underlying 
sky geometry and data type. By default, we assume a two dimensional WCS that is
effectively meaningless for the sky geometry, and a floating point array type.

Slicing Maps
------------

Because the :py:class:`pixell.enmap.ndmap` class is a wrapper around numpy, you
can always slice it in exactly the same way. ``pixell`` will automatically handle
assocating the correct geometry with your cut-out:

.. code-block:: python

    from pixell import enmap
    act_dr6 = enmap.read_map("act_dr6.02_std_AA_day_pa4_f150_4way_coadd_map.fits")

    print(act_dr6)

    cut_out = act_dr6[32:46, 93:99]

    print(cut_out)

However, it's not always that easy!