#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pixell` package."""

import unittest

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

try:                              # when invoked directly...
    import pixel_tests as ptests
except ImportError:               # when imported through py.test
    from . import pixel_tests as ptests

TEST_DIR = ptests.TEST_DIR
DATA_PREFIX = ptests.DATA_PREFIX
lens_version = '091819'

def get_offset_result(res=1.,dtype=np.float64,seed=1):
    shape,wcs  = enmap.fullsky_geometry(res=np.deg2rad(res))
    shape = (3,) + shape
    obs_pos = enmap.posmap(shape, wcs)
    np.random.seed(seed)
    grad = enmap.enmap(np.random.random(shape),wcs)*1e-3
    raw_pos = enmap.samewcs(lensing.offset_by_grad(obs_pos, grad, pol=shape[-3]>1, geodesic=True), obs_pos)
    return obs_pos,grad,raw_pos

def get_lens_result(res=1.,lmax=400,dtype=np.float64,seed=1):
    shape,wcs  = enmap.fullsky_geometry(res=np.deg2rad(res))
    shape = (3,) + shape
    # ells = np.arange(lmax)
    ps_cmb,ps_lens = powspec.read_camb_scalar(DATA_PREFIX+"test_scalCls.dat")
    ps_lensinput = np.zeros((4,4,ps_cmb.shape[-1]))
    ps_lensinput[0,0] = ps_lens
    ps_lensinput[1:,1:] = ps_cmb
    lensed = lensing.rand_map(shape, wcs, ps_lensinput, lmax=lmax, maplmax=None, dtype=dtype, seed=seed, phi_seed=None, oversample=2.0, spin=[0,2], output="lu", geodesic=True, verbose=False, delta_theta=None)
    return lensed

class PixelTests(unittest.TestCase):


    def test_almxfl(self):
        import healpy as hp

        for lmax in [100,400,500,1000]:
            ainfo = sharp.alm_info(lmax)
            alms = hp.synalm(np.ones(lmax+1),lmax = lmax)
            filtering = np.ones(lmax+1)
            alms0 = ainfo.lmul(alms.copy(),filtering)
            assert np.all(np.isclose(alms0,alms))

        for lmax in [100,400,500,1000]:
            ainfo = sharp.alm_info(lmax)
            alms = hp.synalm(np.ones(lmax+1),lmax = lmax)
            alms0 = curvedsky.almxfl(alms.copy(),lambda x: np.ones(x.shape))
            assert np.all(np.isclose(alms0,alms))
            
        

    def test_rand_alm(self):
        def nalm(lmax):
            return (lmax + 1) * (lmax + 2) / 2

        lmaxes = [50, 100, 150, 300]
        
        mypower = np.ones(50)
        for lmax in lmaxes:
            palm = curvedsky.rand_alm(mypower, lmax = lmax)
            halm = curvedsky.rand_alm_healpy(  mypower, lmax = lmax)
            
            print("nalm(%i) = %i, curvedsky.rand_alm gives %s, curvedsky.rand_alm_healpy gives %s "\
	              % (lmax, \
                     nalm(lmax),\
                     palm.shape, \
                     halm.shape)        )
            assert np.all(np.isclose(np.asarray(palm.shape),np.asarray(halm.shape)))
            
    
    def test_offset(self):
        obs_pos,grad,raw_pos = get_offset_result(1.)
        obs_pos0 = enmap.read_map(DATA_PREFIX+"MM_offset_obs_pos_%s.fits" % lens_version)
        grad0 = enmap.read_map(DATA_PREFIX+"MM_offset_grad_%s.fits"  % lens_version)
        raw_pos0 = enmap.read_map(DATA_PREFIX+"MM_offset_raw_pos_%s.fits"  % lens_version)
        assert np.all(np.isclose(obs_pos,obs_pos0))
        assert np.all(np.isclose(raw_pos,raw_pos0))
        assert np.all(np.isclose(grad,grad0))
        assert wcsutils.equal(grad.wcs,grad0.wcs)
        assert wcsutils.equal(obs_pos.wcs,obs_pos0.wcs)
        assert wcsutils.equal(raw_pos.wcs,raw_pos0.wcs)

    def test_lensing(self):
        lensed,unlensed = get_lens_result(1.,400,np.float64)
        lensed0 = enmap.read_map(DATA_PREFIX+"MM_lensed_%s.fits"  % lens_version)
        unlensed0 = enmap.read_map(DATA_PREFIX+"MM_unlensed_%s.fits"  % lens_version)
        y,x = lensed0.posmap()
        assert np.all(np.isclose(lensed,lensed0))
        assert np.all(np.isclose(unlensed,unlensed0))
        assert wcsutils.equal(lensed.wcs,lensed0.wcs)
        assert wcsutils.equal(unlensed.wcs,unlensed0.wcs)
        assert wcsutils.equal(unlensed.wcs,lensed.wcs)
    
    def test_enplot(self):
        print("Testing enplot...")
        shape,wcs = enmap.geometry(pos=(0,0),shape=(3,100,100),res=0.01)
        a = enmap.ones(shape,wcs)
        p = enplot.get_plots(a)

    def test_fft(self):
        # Tests that ifft(ifft(imap))==imap, i.e. default normalizations are consistent
        shape,wcs = enmap.geometry(pos=(0,0),shape=(3,100,100),res=0.01)
        imap = enmap.enmap(np.random.random(shape),wcs)
        assert np.all(np.isclose(imap,enmap.ifft(enmap.fft(imap,normalize='phy'),normalize='phy').real))
        assert np.all(np.isclose(imap,enmap.ifft(enmap.fft(imap)).real))

    def test_extract(self):
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

    def test_fullsky_geometry(self):
        # Tests whether number of pixels and area of a full-sky 0.5 arcminute resolution map are correct
        print("Testing full sky geometry...")
        test_res_arcmin = 0.5
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(test_res_arcmin/60.),proj='car')
        assert shape[0]==21601 and shape[1]==43200
        assert abs(enmap.area(shape,wcs) - 4*np.pi) < 1e-6

    def test_pixels(self):
        """Runs reference pixel and mean-square comparisons on extracts from randomly generated
        maps"""
        print("Testing reference pixels...")
        results,rname = ptests.get_extraction_test_results(TEST_DIR+"/tests.yml")
        cresults = pickle.load(open(DATA_PREFIX+"%s.pkl" % rname,'rb'))
        assert sorted(results.keys())==sorted(cresults.keys())
        for g in results.keys():
            assert sorted(results[g].keys())==sorted(cresults[g].keys())
            for s in results[g].keys():
                assert sorted(results[g][s].keys())==sorted(cresults[g][s].keys())
                for e in results[g][s].keys():
                    assert np.all(np.isclose(results[g][s][e],cresults[g][s][e]))


    def test_sim_slice(self):
        ps = powspec.read_spectrum(DATA_PREFIX+"test_scalCls.dat")[:1,:1]
        test_res_arcmin = 10.0
        lmax = 2000
        fact = 2.
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(test_res_arcmin/60.),proj='car')
        omap = curvedsky.rand_map(shape, wcs, ps,lmax=lmax)
        ofunc = lambda ishape,iwcs: fact*enmap.extract(omap,ishape,iwcs)
        nmap = reproject.populate(shape,wcs,ofunc,maxpixy = 400,maxpixx = 400)
        assert np.all(np.isclose(nmap/omap,2.))


if __name__ == '__main__':
    unittest.main()
    test_sim_slice()
