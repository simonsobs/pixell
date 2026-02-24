Thumbnails and stacking
=======================

A common analysis technique in CMB science is to extract small cutout images
("thumbnails" or "postage stamps") centered on a catalog of objects and then
average them together — a process called *stacking*. This amplifies the signal of
objects that are individually too faint to detect, and reveals the average radial
profile of the signal around that class of object.

Pixell provides :py:func:`pixell.reproject.thumbnails` for this purpose. It
extracts cutouts from a large sky map, optionally reprojects each one onto a local
tangent-plane geometry (removing projection distortions), and handles polarization
rotation automatically.

Extracting thumbnails
----------------------

.. code-block:: python

    from pixell import enmap, reproject, utils
    import numpy as np

    # Load a large sky map
    sky_map = enmap.read_map("my_sky_map.fits")

    # Catalog of objects: shape (N, 2) with columns [dec, ra] in radians
    catalog = np.array([
        [0.00,  0.00],
        [0.05,  0.10],
        [-0.03, 0.07],
    ])

    # Extract 10-arcmin-radius thumbnails at 0.5-arcmin resolution
    thumbs = reproject.thumbnails(
        sky_map,
        catalog,
        r   = 10 * utils.arcmin,   # thumbnail radius
        res = 0.5 * utils.arcmin,  # output pixel resolution
        proj = "tan",              # tangent-plane projection (default "car")
    )

    print(thumbs.shape)   # (3, ny_thumb, nx_thumb)

Each thumbnail is centered on the corresponding catalog position, with the pixel
``(ny//2, nx//2)`` at ``(dec=0, ra=0)`` in the tangent-plane coordinate system
(i.e. the center of the object).

``thumbnails`` automatically apodizes the edges of the extracted region before
interpolation to avoid edge artifacts. The apodization width can be controlled:

.. code-block:: python

    thumbs = reproject.thumbnails(
        sky_map,
        catalog,
        r    = 10 * utils.arcmin,
        res  = 0.5 * utils.arcmin,
        apod = 3 * utils.arcmin,    # apodization width (default 2 arcmin)
    )

Polarized thumbnails
^^^^^^^^^^^^^^^^^^^^^

For T,Q,U maps, ``thumbnails`` rotates Q and U into the local coordinate frame of
each thumbnail so that the E/B decomposition is consistent across the stack:

.. code-block:: python

    # sky_map has shape (3, ny, nx): T, Q, U
    sky_map_pol = enmap.read_map("my_tqu_map.fits")

    thumbs_pol = reproject.thumbnails(
        sky_map_pol,
        catalog,
        r   = 10 * utils.arcmin,
        res = 0.5 * utils.arcmin,
        pol = True,   # rotate Q,U into local frame (auto-detected if shape is (...,3,ny,nx))
    )
    print(thumbs_pol.shape)  # (N, 3, ny_thumb, nx_thumb)

Inverse-variance thumbnails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inverse-variance (ivar) maps require special handling because they are *extensive*
quantities (values scale with pixel area). Use
:py:func:`pixell.reproject.thumbnails_ivar` to handle this correctly:

.. code-block:: python

    from pixell import reproject

    ivar_map = enmap.read_map("my_ivar_map.fits")

    ivar_thumbs = reproject.thumbnails_ivar(
        ivar_map,
        catalog,
        r   = 10 * utils.arcmin,
        res = 0.5 * utils.arcmin,
    )

Stacking
---------

Once thumbnails are extracted, stacking is simply a weighted average:

Unweighted (equal-weight) stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pixell import enmap, reproject, utils
    import numpy as np

    sky_map = enmap.read_map("my_sky_map.fits")

    # Load a catalog of galaxy clusters (N, 2) in radians
    # For this example we simulate random positions
    rng     = np.random.default_rng(0)
    N       = 500
    catalog = rng.uniform(-5, 5, size=(N, 2)) * utils.degree

    thumbs = reproject.thumbnails(
        sky_map, catalog, r=5*utils.arcmin, res=0.5*utils.arcmin
    )

    # Unweighted mean stack
    stack = np.mean(thumbs, axis=0)

    #TODO: add figure -- run code:
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # axes[0].imshow(thumbs[0], origin="lower"); axes[0].set_title("Single thumbnail")
    # axes[1].imshow(stack, origin="lower"); axes[1].set_title("Stack (N=500)")
    # plt.tight_layout(); plt.savefig("stack_result.png", dpi=150)

Inverse-variance weighted stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the noise level varies across the map (e.g. from non-uniform scan coverage),
use inverse-variance weights for an optimal stack:

.. code-block:: python

    from pixell import enmap, reproject, utils
    import numpy as np

    sky_map  = enmap.read_map("my_sky_map.fits")
    ivar_map = enmap.read_map("my_ivar_map.fits")

    catalog = np.array([...])   # (N, 2) in radians

    # Extract map and ivar thumbnails
    thumbs      = reproject.thumbnails(sky_map,  catalog, r=5*utils.arcmin, res=0.5*utils.arcmin)
    ivar_thumbs = reproject.thumbnails_ivar(ivar_map, catalog, r=5*utils.arcmin, res=0.5*utils.arcmin)

    # Weighted sum
    stack_num = np.sum(thumbs * ivar_thumbs, axis=0)
    stack_den = np.sum(ivar_thumbs, axis=0)
    stack_ivar = stack_den  # ivar of the stack

    # Avoid division by zero
    stack = np.where(stack_den > 0, stack_num / stack_den, 0.0)

Radial profile from a stack
-----------------------------

After stacking, you can extract a radial profile using :py:func:`pixell.enmap.rbin`:

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    # stack is an ndmap (the result of thumbnails has a WCS)
    # center on the middle pixel
    rs, profile = enmap.rbin(stack, bsize=0.5 * utils.arcmin)

    #TODO: add figure -- run code:
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(6, 4))
    # plt.plot(np.rad2deg(rs) * 60, profile)
    # plt.xlabel("Angular radius (arcmin)")
    # plt.ylabel("Stacked signal (uK)")
    # plt.title("Radial profile from stack")
    # plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout(); plt.savefig("stack_radial_profile.png", dpi=150)

Simple pixel-index stamps
--------------------------

For quick cutouts that do not need reprojection, use
:py:func:`pixell.enmap.stamps`, which extracts a fixed pixel box around each
position. The output geometry still has a valid WCS:

.. code-block:: python

    from pixell import enmap, utils
    import numpy as np

    m = enmap.read_map("my_map.fits")

    # Positions in radians
    pos = np.array([[0.0, 0.0], [0.05, 0.1]])   # (N, 2) -- (dec, ra)

    # Extract 61x61 pixel stamps
    stamps = m.stamps(pos, shape=(61, 61))
    print(stamps.shape)   # (2, 61, 61) for a single-component map

    # Or as a list of enmaps (useful when they may fall off the edge)
    stamps_list = m.stamps(pos, shape=(61, 61), aslist=True)
