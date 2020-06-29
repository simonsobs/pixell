=======
History
=======

0.1.0 (2018-06-15)
------------------

* First release on PyPI.

0.5.2 (2019-01-22)
------------------

* API for most modules is close to converged
* Significant number of bug fixes and new features
* Versioning system implemented through versioneer and bumpversion
* Automated pixel level tests for discovering effects of low-level changes

0.6.0 (2019-09-18)
------------------

Changes relative to 0.5.2 include:

* Improvements in accuracy for map extent, area and Fourier wavenumbers
* Spherical harmonic treatment consistent with healpy
* Additional helper functions, e.g enmap.insert
* Helper arguments, e.g. physical normalization for enmap.fft
* Bug fixes e.g. in rand_alm
* Improved installation procedure and documentation

0.9.6 (2020-06-22)
------------------

Changes relative to 0.6.0 include:

* Ability to read compressed FITS images
* Fixed a bug to make aberration and modulation accurate to all orders
* Expanded alm2cl to handle full cross-spectra and broadcasting

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

0.10.3 (2020-06-26)
-------------------

Changes relative to 0.10.2 include:

* Bug fix for automatic IAU -> COSMO, recognizes POLCCONV instead of POLCONV.
