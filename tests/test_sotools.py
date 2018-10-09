#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pixell` package."""


from pixell import enmap
from pixell import sharp
from pixell import curvedsky
from pixell import lensing,interpol
import numpy as np
import pickle
import os,sys

def test_fullsky_geometry():
    # Tests whether number of pixels and area of a full-sky 0.5 arcminute resolution map are correct
    test_res_arcmin = 0.5
    shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(test_res_arcmin/60.),proj='car')
    assert shape[0]==21601 and shape[1]==43200
    assert 50000 < (enmap.area(shape,wcs)*(180./np.pi)**2.) < 51000

def test_pixels():
    """Runs reference pixel and mean-square comparisons on extracts from randomly generated
    maps"""
    from . import pixel_tests as ptests
    path = os.path.dirname(enmap.__file__)+"/../tests/"
    results,rname = ptests.get_extraction_test_results(path+"tests.yml")
    cresults = pickle.load(open(path+"data/%s.pkl" % rname,'rb'))
    assert sorted(results.keys())==sorted(cresults.keys())
    for g in results.keys():
        assert sorted(results[g].keys())==sorted(cresults[g].keys())
        for s in results[g].keys():
            assert sorted(results[g][s].keys())==sorted(cresults[g][s].keys())
            for e in results[g][s].keys():
                assert np.all(np.isclose(results[g][s][e],cresults[g][s][e]))
