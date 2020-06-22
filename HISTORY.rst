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
