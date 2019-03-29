#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pixell` package."""


from pixell import sharp
from pixell import enmap
from pixell import curvedsky
from pixell import lensing
from pixell import interpol
from pixell import array_ops
from pixell import enplot
from pixell import powspec
from pixell import reproject
from pixell import wcsutils
import numpy as np
import pickle
import os,sys

def test_enplot():
    print("Testing enplot...")
    shape,wcs = enmap.geometry(pos=(0,0),shape=(3,100,100),res=0.01)
    a = enmap.ones(shape,wcs)
    p = enplot.get_plots(a)

def test_fft():
    # Tests that ifft(ifft(imap))==imap, i.e. default normalizations are consistent
    shape,wcs = enmap.geometry(pos=(0,0),shape=(3,100,100),res=0.01)
    imap = enmap.enmap(np.random.random(shape),wcs)
    assert np.all(np.isclose(imap,enmap.ifft(enmap.fft(imap,normalize='phy'),normalize='phy').real))
    assert np.all(np.isclose(imap,enmap.ifft(enmap.fft(imap)).real))

def test_extract():
    # Tests that extraction is sensible
    shape,wcs = enmap.geometry(pos=(0,0),shape=(500,500),res=0.01)
    imap = enmap.enmap(np.random.random(shape),wcs)
    smap = imap[200:300,200:300]
    sshape,swcs = smap.shape,smap.wcs
    smap2 = enmap.extract(imap,sshape,swcs)
    pixbox = enmap.pixbox_of(imap.wcs,sshape,swcs)
    # Do write and read test
    filename = "temporary_extract_map.fits" # NOT THREAD SAFE
    enmap.write_map(filename,imap)
    smap3 = enmap.read_map(filename,pixbox=pixbox)
    os.remove(filename)
    assert np.all(np.isclose(smap,smap2))
    assert np.all(np.isclose(smap,smap3))
    assert wcsutils.equal(smap.wcs,smap2.wcs)
    assert wcsutils.equal(smap.wcs,smap3.wcs)
    
    

def test_fullsky_geometry():
    # Tests whether number of pixels and area of a full-sky 0.5 arcminute resolution map are correct
    print("Testing full sky geometry...")
    test_res_arcmin = 0.5
    shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(test_res_arcmin/60.),proj='car')
    assert shape[0]==21601 and shape[1]==43200
    assert 50000 < (enmap.area(shape,wcs)*(180./np.pi)**2.) < 51000

def test_pixels():
    """Runs reference pixel and mean-square comparisons on extracts from randomly generated
    maps"""
    from . import pixel_tests as ptests
    print("Testing reference pixels...")
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


def test_sim_slice():
    path = os.path.dirname(enmap.__file__)+"/../tests/"
    ps = powspec.read_spectrum(path+"data/test_scalCls.dat")[:1,:1]
    test_res_arcmin = 10.0
    lmax = 2000
    fact = 2.
    shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(test_res_arcmin/60.),proj='car')
    omap = curvedsky.rand_map(shape, wcs, ps,lmax=lmax)
    ofunc = lambda ishape,iwcs: fact*enmap.extract(omap,ishape,iwcs)
    nmap = reproject.populate(shape,wcs,ofunc,maxpixy = 400,maxpixx = 400)
    assert np.all(np.isclose(nmap/omap,2.))
