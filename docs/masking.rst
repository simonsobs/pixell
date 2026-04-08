Masking
=======

Masks are boolean or floating-point ``ndmap`` objects that indicate which pixels
should be included in (or excluded from) an analysis.  In ``pixell``, a mask
conventionally has value ``1`` in good pixels and ``0`` in bad pixels, though
many operations work with any weighting map.

Everything below assumes::

   from pixell import enmap, utils
   import numpy as np


Creating masks
--------------

Threshold masking
^^^^^^^^^^^^^^^^^

The simplest mask is a threshold applied to a map.  For example, to mask pixels
above a noise level:

.. code-block:: python

   >>> mask = (noise_map < threshold).astype(float)

Since ``ndmap`` inherits numpy comparisons, the result is already an ``ndmap``
with the correct WCS.

To mask pixels with no coverage (e.g. a hits map):

.. code-block:: python

   >>> mask = (hits_map > 0).astype(float)

Geometric masks
^^^^^^^^^^^^^^^

:py:func:`pixell.enmap.distance_from` computes the distance from each pixel to the
nearest member of a set of sky positions.  This is useful for masking around known
point sources:

.. code-block:: python

   >>> # pos has shape (2, N) with rows [dec, ra] in radians
   >>> dist  = enmap.distance_from(shape, wcs, pos)
   >>> mask  = (dist > 5*utils.arcmin).astype(float)

The ``rmax`` argument limits the search radius and can speed things up significantly
when masking a large number of point sources over a small fraction of the map.

For a single centre point (e.g. a circular aperture mask):

.. code-block:: python

   >>> centre = np.array([0.0, 0.0])      # (dec, ra) in radians
   >>> dist   = imap.modrmap(ref=centre)
   >>> disk   = (dist < 30*utils.arcmin).astype(float)

Distance transform
^^^^^^^^^^^^^^^^^^

Given a binary mask (``True`` where pixels are *good*),
:py:func:`pixell.enmap.distance_transform` returns, for every pixel, the angular
distance to the nearest *bad* (``False``) pixel.  This is useful for constructing
masks that transition smoothly away from a survey edge:

.. code-block:: python

   >>> # binary_mask is True in good pixels
   >>> dist_from_edge = enmap.distance_transform(binary_mask)

The ``rmax`` argument caps the maximum distance computed, which saves time if you
only need distances out to some maximum.

The ``labeled_distance_transform`` variant also returns the label of the nearest
bad region, which can be used to assign each good pixel to its closest boundary
segment.


Apodisation
-----------

Hard-edged masks introduce ringing in Fourier or harmonic transforms.
Apodisation smooths the transition to zero at mask boundaries to suppress this.
``pixell`` provides two complementary functions.

Tapering the map boundary with ``apod``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:func:`pixell.enmap.apod` tapers a map to zero at its rectangular boundary over
a given number of pixels:

.. code-block:: python

   >>> imap_apo = enmap.apod(imap, width=60)    # taper over 60 pixels

The ``profile`` argument selects the taper shape.  ``"cos"`` (default) gives a
raised-cosine (Hann) window; ``"lin"`` gives a linear ramp.  The ``fill`` argument
sets the target value: ``"zero"`` (default) tapers to zero, while ``"mean"`` first
subtracts the map mean so the taper is towards the mean rather than zero:

.. code-block:: python

   >>> imap_apo = enmap.apod(imap, width=60, fill="mean")

Pass ``inplace=True`` to modify the map in place rather than returning a copy.

When computing a power spectrum from an apodised map, divide by the mean-square
taper to correct for the lost power (the W\ :sub:`2` factor)::

   taper  = enmap.apod(enmap.ones(shape, wcs), width=60)
   W2     = np.mean(taper**2)
   imap_apo = imap * taper
   kmap   = enmap.fft(imap_apo, normalize="phys")
   cl     = np.abs(kmap)**2 / W2

Tapering from a custom mask with ``apod_mask``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you have an irregular survey footprint, :py:func:`pixell.enmap.apod_mask`
turns a binary mask into a smooth apodisation window that tapers from zero in bad
pixels to one deep inside the good region:

.. code-block:: python

   >>> # binary_mask is 1 inside the survey footprint, 0 outside
   >>> apo_window = enmap.apod_mask(binary_mask, width=0.5*utils.degree)

The ``width`` argument (in radians) controls how quickly the mask ramps up from
zero.  Regions outside the map boundary are automatically treated as bad.  The
transition profile defaults to a raised cosine.

The resulting ``apo_window`` ranges from 0 to 1 and can be multiplied directly
onto a map before transforming::

   imap_apo = imap * apo_window


Combining masks
---------------

Masks can be combined with standard numpy logical operations:

.. code-block:: python

   >>> # Pixels that pass both masks
   >>> combined = mask1 * mask2
   >>> # Pixels that pass either mask
   >>> union    = np.maximum(mask1, mask2)
   >>> # Invert a mask
   >>> inverted = 1 - mask

When masks have been apodised (floating-point values between 0 and 1), use
multiplication to combine them; this is equivalent to an AND operation that
respects the smooth transitions.


Applying a mask
---------------

Multiply the mask onto the map directly:

.. code-block:: python

   >>> imap_masked = imap * mask

To set masked pixels to NaN rather than zero (useful for some downstream tools):

.. code-block:: python

   >>> imap_nan = imap.copy()
   >>> imap_nan[mask == 0] = np.nan

To fill NaN or zero pixels back in with the map mean::

   imap.fillbad(val=0)
