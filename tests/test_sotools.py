#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sotools` package."""


from sotools import enmap
from sotools import sharp
from sotools import curvedsky
import numpy as np

def test_fullsky_geometry():
    # Tests whether number of pixels and area of a full-sky 0.5 arcminute resolution map are correct
    test_res_arcmin = 0.5
    shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(test_res_arcmin/60.),proj='car')
    assert shape[0]==21601 and shape[1]==43200
    assert 50000 < (enmap.area(shape,wcs)*(180./np.pi)**2.) < 51000

