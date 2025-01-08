=======
History
=======

0.26.2 (2024-09-24)
-------------------

Changes relative to 0.21.0 include:

* Significant changes to build and deployment system, now based on meson
* Improvements to sky aberration and modulation
* Minor bug fixes in reproject and curvedsky 
* Improvements to wavelet analysis


0.21.0 (2023-10-19)
-------------------

Changes relative to 0.19.2 include:

* More miscellaneous fixes after transition to ducc0
* More work on wavelets
* We now provide MacOS x86_64 wheels
* Improved build system that respects choices of CC, CXX, and FC

0.19.2 (2023-08-18)
-------------------

Changes relative to 0.19.0 include:

* Important bugfixes for the migration from libsharp2 to ducc0
* Improved SHT unit tests


0.19.0 (2023-07-14)
-------------------

Changes relative to 0.17.3 include:

* Migrate fully from libsharp2 to ducc0 (for curved sky functions)
* Temporary suspension of MacOS pip binaries (use `pip install pixell==0.17.3` for Macs in the meantime)
* Miscellaneous fixes

0.17.3 (2023-03-17)
-------------------

Changes relative to 0.17.2 include:

* More support for fejer1
  
0.17.2 (2023-02-21)
-------------------

Changes relative to 0.17.1 include:

* Build for Python 3.11

0.17.1 (2023-01-26)
-------------------

Changes relative to 0.16.0 include:

* Bilinear map-making pixel window function
* Miscellaneous new functions and API improvements
* Miscellaneous bug fixes
* Fixes for Apple Silicon



0.16.0 (2022-06-08)
-------------------

Changes relative to 0.15.3 include:

* Wavelet analysis
* Fast C-based source simulation
* Fast vectorized radial profile binning
* Fixes and improvements to HDF5 I/O
* Fixes to OSX support


0.15.3 (2022-02-12)
-------------------

Changes relative to 0.15.1 include:

* New wheels that fix numpy binary incompatibility errors


0.15.1 (2022-01-23)
-------------------

Changes relative to 0.14.3 include:

* More flexible enmap.read_map_geometry
* Add Python 3.10 support, drop Python 3.6 support

0.14.3 (2021-12-13)
-------------------

Changes relative to 0.14.2 include:

* Updates to enmap.insert, UHTs and WCS string accuracy

0.14.2 (2021-11-23)
-------------------

Changes relative to 0.14.1 include:

* An important bugfix for enmap.downgrade when the `op` argument is passed. This bug has been present since v0.14.0 and in commits on master since Aug 12, 2021.

0.14.1 (2021-11-16)
-------------------

Changes relative to 0.13.2 include:

* A breaking change to map2alm where it no longer approximates WCS if ring weights are unavailable
* Miscellaneous bug fixes
* ducc0 FFT support and fast rotate_alm
* Tiled map support
* New healpix <-> rectpix reprojection API


0.13.2 (2021-07-16)
-------------------

Changes relative to 0.13.1 include:

* Added binaries for MacOS 11 Big Sur

0.13.1 (2021-07-08)
-------------------

Changes relative to 0.13.0 include:

* Fixes to the MacOS wheel building


0.13.0 (2021-07-08)
-------------------

Changes relative to 0.12.1 include:

* Matched filtering in a new analysis module
* Conjugate gradients solver
* Discrete cosine transforms
* Miscellaneous bug fixes
  

0.12.1 (2021-04-30)
-------------------

Changes relative to 0.12.0 include:

* Patch to fix numpy binary incompatibility issues
  caused by changes to the numpy C API. We now require
  numpy >1.20.


0.12.0 (2021-04-13)
-------------------

Changes relative to 0.11.2 include:

* We now use libsharp2 instead of libsharp, which has signficantly faster SHTs
* Major breaking change: the meaning of the "iau" flag has been
  corrected and reversed. The default behaviour of map2harm and other functions
  using this flag will be different.
* Unified harmonic transforms module
* postage_stamp removed in favor of thumbnails
* Adjoint harmonic transforms
  
0.11.2 (2021-02-04)
-------------------

Changes relative to 0.11.0 include:

* Bug-fix for distance_transform when using rmax


0.11.0 (2021-02-02)
-------------------

Changes relative to 0.10.3 include:

* Bug-fix for enmap.project that led to crashes
* enplot improvements
* Improvements to fft and ifft overhead
* alm filtering API improvements
* Changes to CMB dipole parameter
* Allow lmax!=mmax in curvedsky routines
* Python 3.9 builds and Github actions instead of Travis


0.10.3 (2020-06-26)
-------------------

Changes relative to 0.10.2 include:

* Bug fix for automatic IAU -> COSMO, recognizes POLCCONV instead of POLCONV.

0.10.2 (2020-06-26)
-------------------

Changes relative to 0.9.6 include:

* Automatically converts maps recognized to be in IAU polarization convention
  (through the FITS header) to COSMO convention by flipping the sign of U
* Fixes a centering issue in reproject.thumbnails
* Optimizes posmap for separable projections and pixsizemap for cylindrical
  projections making these functions orders of magnitude faster for CAR (and
  other projections)
* A test script test-pixell is distributed with the package

0.9.6 (2020-06-22)
------------------

Changes relative to 0.6.0 include:

* Ability to read compressed FITS images
* Fixed a bug to make aberration and modulation accurate to all orders
* Expanded alm2cl to handle full cross-spectra and broadcasting

0.6.0 (2019-09-18)
------------------

Changes relative to 0.5.2 include:

* Improvements in accuracy for map extent, area and Fourier wavenumbers
* Spherical harmonic treatment consistent with healpy
* Additional helper functions, e.g enmap.insert
* Helper arguments, e.g. physical normalization for enmap.fft
* Bug fixes e.g. in rand_alm
* Improved installation procedure and documentation


0.5.2 (2019-01-22)
------------------

* API for most modules is close to converged
* Significant number of bug fixes and new features
* Versioning system implemented through versioneer and bumpversion
* Automated pixel level tests for discovering effects of low-level changes

  
0.1.0 (2018-06-15)
------------------

* First release on PyPI.

