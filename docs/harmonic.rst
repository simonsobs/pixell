Spherical harmonic analysis
===========================

Working with spherical harmonic transforms (SHTs) is common in CMB analysis,
since maps physically live on the sphere. There are notorious subtleties
associated with using SHTs, compared to say Fourier transforms (FFTs), but
fortunately ``pixell`` makes things easy and transparent. In addition to the
basics, this guide will cover some more advanced topics that you may encounter.

From maps to harmonics
----------------------
An SHT is analagous to an FFT, but for maps that live on the sphere rather than
the flat sky (as ours do). Their output is typically referred to as an ``alm``,
which are the components of the input map in the
`spherical harmonic basis <https://en.wikipedia.org/wiki/Spherical_harmonics#Spherical_harmonics_expansion>`_
(i.e., this is read as a vector *a* with values indexed by spherical harmonic
mode indices *l* (inverse angular scale) and *m* (azimuthal position)). 
Conventionally, each ``alm`` is represented in memory as a 1d vector: all of the
2d information in the map gets reshaped into a 1d list of numbers in spherical
harmonic space. If the map has a polarization component
(so, a shape like ``(3, Ny, Nx)``), then the output shape will be ``(3, nalm)``.
This is just a convention and can be controlled in ``pixell``, as we will see!

You typically will use :py:func:`curvedsky.map2alm` to compute the SHT of a map.
Unlike an FFT, where the output format is exactly fixed by the input, we need to
give an SHT more information. Here, we give it ``lmax``, the maximum multipole
(smallest physical scale) to be computed:

.. code-block:: python

    from pixell import enmap, curvedsky

    # first load the map so we have something to work with
    imap = enmap.read_map("act_dr6.02_std_AA_day_pa4_f150_4way_coadd_map.fits")

    # The transformation is limited by the `lmax` factor. Increasing this will make
    # the alms exact on smaller scales but will also result in slower run times
    lmax = 4000
    alm = curvedsky.map2alm(imap[0], lmax=lmax)

The first argument to ``curvedsky.map2alm`` 