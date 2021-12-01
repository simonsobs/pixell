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
from pixell import pointsrcs
from pixell import wcsutils
from pixell import utils as u
from pixell import colors
from pixell import fft
from pixell import tilemap
from pixell import utils
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
            alms = hp.synalm(np.ones(lmax+1),lmax = lmax, new=True)
            filtering = np.ones(lmax+1)
            alms0 = ainfo.lmul(alms.copy(),filtering)
            assert np.all(np.isclose(alms0,alms))

        for lmax in [100,400,500,1000]:
            ainfo = sharp.alm_info(lmax)
            alms = hp.synalm(np.ones(lmax+1),lmax = lmax, new=True)
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

    def test_fft_input_shape(self):
        # Tests fft for various shapes and choices of axes.
        # 1D FFT over last axis for 3d array.
        signal = np.ones((1, 2, 5))
        signal[0,1,:] = 10.
        out_exp = np.zeros((1, 2, 5), dtype=np.complex128)
        out_exp[0,0,0] = 5
        out_exp[0,1,0] = 50
        out = fft.fft(signal)
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 1D FFT over middle axis for 3d array.
        signal = np.ones((1, 5, 2))
        signal[0,:,1] = 10.
        out_exp = np.zeros((1, 5, 2), dtype=np.complex128)
        out_exp[0,0,0] = 5
        out_exp[0,0,1] = 50
        out = fft.fft(signal, axes=[-2])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 2D FFT over last 2 axes of 4d array.
        signal = np.ones((1, 2, 5, 10))
        signal[0,1,:] = 10.
        out_exp = np.zeros((1, 2, 5, 10), dtype=np.complex128)
        out_exp[0,0,0,0] = 50
        out_exp[0,1,0,0] = 500
        out = fft.fft(signal, axes=[-2, -1])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 2D FFT over last 2 axes of 4d non-contiguous array.
        signal = np.ones((1, 2, 5, 10), dtype=np.complex128)
        signal[0,1,:] = 10
        ft = np.zeros((5, 10, 1, 2), dtype=np.complex128).transpose(2, 3, 0, 1)
        out_exp = np.zeros_like(ft)
        out_exp[0,0,0,0] = 50
        out_exp[0,1,0,0] = 500
        out = fft.fft(signal, ft=ft, axes=[-2, -1])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(np.shares_memory(ft, out))
        self.assertFalse(out.flags['C_CONTIGUOUS'])

        # 2D FFT over middle 2 axes of 4d array.
        signal = np.ones((1, 5, 10, 2))
        signal[0,:,:,1] = 10.
        out_exp = np.zeros((1, 5, 10, 2), dtype=np.complex128)
        out_exp[0,0,0,0] = 50
        out_exp[0,0,0,1] = 500
        out = fft.fft(signal, axes=[-3, -2])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

    def test_ifft_input_shape(self):
        # Tests ifft for various shapes and choices of axes.
        # 1D IFFT over last axis for 3d array.
        fsignal = np.ones((1, 2, 5), dtype=np.complex128)
        fsignal[0,1,:] = 10.
        out_exp = np.zeros((1, 2, 5))
        out_exp[0,0,0] = 5
        out_exp[0,1,0] = 50
        out = fft.ifft(fsignal)
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 1D IFFT over middle axis for 3d array.
        fsignal = np.ones((1, 5, 2), dtype=np.complex128)
        fsignal[0,:,1] = 10.
        out_exp = np.zeros((1, 5, 2))
        out_exp[0,0,0] = 5
        out_exp[0,0,1] = 50
        out = fft.ifft(fsignal, axes=[-2])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 2D IFFT over last 2 axes of 4d array.
        fsignal = np.ones((1, 2, 5, 10), dtype=np.complex128)
        fsignal[0,1,:] = 10.
        out_exp = np.zeros((1, 2, 5, 10))
        out_exp[0,0,0,0] = 50
        out_exp[0,1,0,0] = 500
        out = fft.ifft(fsignal, axes=[-2, -1])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 2D IFFT over last 2 axes of 4d non-contiguous array.
        fsignal = np.ones((1, 2, 5, 10), dtype=np.complex128)
        fsignal[0,1,:] = 10.
        tod = np.zeros((5, 10, 1, 2), dtype=np.complex128).transpose(2, 3, 0, 1)
        out_exp = np.zeros_like(tod)
        out_exp[0,0,0,0] = 50
        out_exp[0,1,0,0] = 500
        out = fft.ifft(fsignal, tod=tod, axes=[-2, -1])
        self.assertTrue(np.shares_memory(tod, out))
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertFalse(out.flags['C_CONTIGUOUS'])

        # 2D IFFT over middle 2 axes of 4d array.
        fsignal = np.ones((1, 5, 10, 2), dtype=np.complex128)
        fsignal[0,:,:,1] = 10.
        out_exp = np.zeros((1, 5, 10, 2))
        out_exp[0,0,0,0] = 50
        out_exp[0,0,0,1] = 500
        out = fft.ifft(fsignal, axes=[-3, -2])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

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


    def test_b_sign(self):
        """
        We generate a random IQU map with geometry such that cdelt[0]<0
        We transform this to TEB with map2harm and map2alm followed by 
        scalar harm2map and alm2map and use these as reference T,E,B maps.
        We flip the original map along the RA direction.
        We transform this to TEB with map2harm and map2alm followed by 
        scalar harm2map and alm2map and use these as comparison T,E,B maps.
        We compare these maps.
        """
        ells,cltt,clee,clbb,clte = np.loadtxt(DATA_PREFIX+"cosmo2017_10K_acc3_lensedCls.dat",unpack=True)
        ps_cmb = np.zeros((3,3,ells.size))
        ps_cmb[0,0] = cltt
        ps_cmb[1,1] = clee
        ps_cmb[2,2] = clbb
        ps_cmb[1,0] = clte
        ps_cmb[0,1] = clte
        np.random.seed(100)

        # Curved-sky is fine
        lmax = 1000
        alm = curvedsky.rand_alm_healpy(ps_cmb,lmax=lmax)
        shape,iwcs = enmap.fullsky_geometry(res=np.deg2rad(10./60.))
        wcs = enmap.empty(shape,iwcs)[...,::-1].wcs
        shape = (3,) + shape
        imap = curvedsky.alm2map(alm,enmap.empty(shape,wcs))
        oalm = curvedsky.map2alm(imap.copy(),lmax=lmax)
        rmap = curvedsky.alm2map(oalm,enmap.empty(shape,wcs),spin=0)

        imap2 = imap.copy()[...,::-1]
        oalm = curvedsky.map2alm(imap2.copy(),lmax=lmax)
        rmap2 = curvedsky.alm2map(oalm,enmap.empty(shape,wcs),spin=0)

        assert np.all(np.isclose(rmap[0],rmap2[0]))
        assert np.all(np.isclose(rmap[1],rmap2[1]))
        assert np.all(np.isclose(rmap[2],rmap2[2]))
        

        # Flat-sky
        px = 2.0
        N = 300
        shape,iwcs = enmap.geometry(pos=(0,0),res=np.deg2rad(px/60.),shape=(300,300))
        shape = (3,) + shape
        a = enmap.zeros(shape,iwcs)
        a = a[...,::-1]
        wcs = a.wcs

        seed = 100
        imap = enmap.rand_map(shape,wcs,ps_cmb,seed=seed)
        kmap = enmap.map2harm(imap.copy())
        rmap = enmap.harm2map(kmap,spin=0) # reference map

        imap = imap[...,::-1]
        kmap = enmap.map2harm(imap.copy())
        rmap2 = enmap.harm2map(kmap,spin=0)[...,::-1] # comparison map
        
        assert np.all(np.isclose(rmap[0],rmap2[0]))
        assert np.all(np.isclose(rmap[1],rmap2[1],atol=1e0))
        assert np.all(np.isclose(rmap[2],rmap2[2],atol=1e0))

    def test_plain_wcs(self):
        # Test area and box for a small Cartesian geometry
        shape,wcs = enmap.geometry(res=np.deg2rad(1./60.),shape=(600,600),pos=(0,0),proj='plain')
        box = np.rad2deg(enmap.box(shape,wcs))
        area = np.rad2deg(np.rad2deg(enmap.area(shape,wcs)))
        assert np.all(np.isclose(box,np.array([[-5,-5],[5,5]])))
        assert np.isclose(area,100.)

        # and for an artifical Cartesian geometry with area>4pi
        shape,wcs = enmap.geometry(res=np.deg2rad(10),shape=(100,100),pos=(0,0),proj='plain')
        box = np.rad2deg(enmap.box(shape,wcs))
        area = np.rad2deg(np.rad2deg(enmap.area(shape,wcs)))
        assert np.all(np.isclose(box,np.array([[-500,-500],[500,500]])))
        assert np.isclose(area,1000000)


    def test_pospix(self):
        # Posmap separable and non-separable on CAR
        for res in [6,12,24]:
            shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj='car')
            posmap1 = enmap.posmap(shape,wcs)
            posmap2 = enmap.posmap(shape,wcs,separable=True)
            assert np.all(np.isclose(posmap1,posmap2))

        # Pixmap plain
        pres = 0.5
        shape,wcs = enmap.geometry(pos=(0,0),shape=(30,30),res=pres*u.degree,proj='plain')
        yp,xp = enmap.pixshapemap(shape,wcs)
        assert np.all(np.isclose(yp,pres*u.degree))
        assert np.all(np.isclose(xp,pres*u.degree))
        yp,xp = enmap.pixshape(shape,wcs)
        parea = enmap.pixsize(shape,wcs)
        assert np.isclose(parea,(pres*u.degree)**2)
        assert np.isclose(yp,pres*u.degree)
        assert np.isclose(xp,pres*u.degree)
        pmap = enmap.pixsizemap(shape,wcs)
        assert np.all(np.isclose(pmap,(pres*u.degree)**2))

        # Pixmap CAR
        pres = 0.1
        dec_cut = 89.5 # pixsizemap is not accurate near the poles currently
        shape,wcs = enmap.band_geometry(dec_cut=dec_cut*u.degree,res=pres*u.degree,proj='car')
        # Current slow and general but inaccurate near the poles implementation
        pmap = enmap.pixsizemap(shape,wcs)
        # Fast CAR-specific pixsizemap implementation
        dra, ddec = wcs.wcs.cdelt*u.degree
        dec = enmap.posmap([shape[-2],1],wcs)[0,:,0]
        area = np.abs(dra*(np.sin(np.minimum(np.pi/2.,dec+ddec/2))-np.sin(np.maximum(-np.pi/2.,dec-ddec/2))))
        Nx = shape[-1]
        pmap2 = enmap.ndmap(area[...,None].repeat(Nx,axis=-1),wcs)
        assert np.all(np.isclose(pmap,pmap2))
        

    def test_project_nn(self):
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(12/60.),proj='car')
        shape2,wcs2 = enmap.fullsky_geometry(res=np.deg2rad(6/60.),proj='car')
        shape3,wcs3 = enmap.fullsky_geometry(res=np.deg2rad(24/60.),proj='car')
        imap = enmap.ones(shape,wcs)
        omap2 = enmap.project(imap,shape2,wcs2,order=0,mode='wrap')
        omap3 = enmap.project(imap,shape3,wcs3,order=0,mode='wrap')
        assert np.all(np.isclose(omap2,1))
        assert np.all(np.isclose(omap3,1))

    def test_wcsunequal(self):
        shape1,wcs1 = enmap.geometry(pos=(0,0),shape=(100,100),res=1*u.arcmin,proj='car')
        shape1,wcs2 = enmap.geometry(pos=(0,0),shape=(100,100),res=1*u.arcmin,proj='cea')
        shape1,wcs3 = enmap.geometry(pos=(10,10),shape=(100,100),res=1*u.arcmin,proj='car')
        shape1,wcs4 = enmap.geometry(pos=(0,0),shape=(100,100),res=2*u.arcmin,proj='car')
        assert not(wcsutils.equal(wcs1,wcs2))
        assert not(wcsutils.equal(wcs1,wcs3))
        assert not(wcsutils.equal(wcs1,wcs4))
        
        
    def test_scale(self):
        # Test (with a plain geometry) that scale_geometry
        # will result in geometries with the same bounding box
        # but different area pixel
        pres = 0.5
        ufact = 2
        dfact = 0.5
        shape,wcs = enmap.geometry(pos=(0,0),shape=(30,30),res=pres*u.arcmin,proj='plain')
        ushape,uwcs = enmap.scale_geometry(shape,wcs,ufact)
        dshape,dwcs = enmap.scale_geometry(shape,wcs,dfact)
        box = enmap.box(shape,wcs)
        ubox = enmap.box(ushape,uwcs)
        dbox = enmap.box(dshape,dwcs)
        parea = enmap.pixsize(shape,wcs)
        uparea = enmap.pixsize(ushape,uwcs)
        dparea = enmap.pixsize(dshape,dwcs)
        assert np.all(np.isclose(box,ubox))
        assert np.all(np.isclose(box,dbox))
        assert np.isclose(parea/(ufact**2),uparea)
        assert np.isclose(parea/(dfact**2),dparea)

    def test_prepare_alm_mmax(self):
        # Check if mmax is correctly handled by prepare_alm.

        # Create lmax=mmax=3 alm array and corresponding alm_info.
        lmax = 3
        nalm = 10  # Triangular alm array of lmax=3 has 10 elements.
        alm_in = np.arange(nalm, dtype=np.complex128)
        ainfo_in = sharp.alm_info(
            lmax=3, mmax=3, nalm=nalm, stride=1, layout="triangular")

        # Case 1: provide only alm.
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=alm_in, ainfo=None)

        np.testing.assert_array_almost_equal(alm_out, alm_in)
        self.assertIs(ainfo_out.lmax, ainfo_in.lmax)
        self.assertIs(ainfo_out.mmax, ainfo_in.mmax)
        self.assertIs(ainfo_out.nelem, ainfo_in.nelem)

        # Case 2: provide only alm_info.
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=None, ainfo=ainfo_in)
        # Expect zero array.
        np.testing.assert_array_almost_equal(alm_out, alm_in * 0)
        self.assertIs(ainfo_out.lmax, ainfo_in.lmax)
        self.assertIs(ainfo_out.mmax, ainfo_in.mmax)
        self.assertIs(ainfo_out.nelem, ainfo_in.nelem)

        # Case 3: provide alm and alm_info
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=alm_in, ainfo=ainfo_in)

        np.testing.assert_array_almost_equal(alm_out, alm_in)
        self.assertIs(ainfo_out.lmax, ainfo_in.lmax)
        self.assertIs(ainfo_out.mmax, ainfo_in.mmax)
        self.assertIs(ainfo_out.nelem, ainfo_in.nelem)

        # Case 4: provide only alm with lmax=3 and mmax=1.
        # This should currently fail.
        nalm = 7
        alm_in = np.arange(7, dtype=np.complex128)
        self.assertRaises(AssertionError, curvedsky.prepare_alm,
                          **dict(alm=alm_in, ainfo=None, lmax=lmax))

        # Case 5: provide only alm_info with lmax=3 and mmax=1.
        nalm = 7
        ainfo_in = sharp.alm_info(
            lmax=3, mmax=1, nalm=nalm, stride=1, layout="triangular")
        alm_exp = np.zeros(7, dtype=np.complex128)
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=None, ainfo=ainfo_in)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)
        self.assertIs(ainfo_out.lmax, ainfo_in.lmax)
        self.assertIs(ainfo_out.mmax, ainfo_in.mmax)
        self.assertIs(ainfo_out.nelem, ainfo_in.nelem)

        # Case 6: provide both alm and alm_info with lmax=3 and mmax=1.
        # This should be allowed.
        nalm = 7
        ainfo_in = sharp.alm_info(
            lmax=3, mmax=1, nalm=nalm, stride=1, layout="triangular")
        alm_in = np.arange(7, dtype=np.complex128)
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=alm_in, ainfo=ainfo_in)

        np.testing.assert_array_almost_equal(alm_out, alm_in)
        self.assertIs(ainfo_out.lmax, ainfo_in.lmax)
        self.assertIs(ainfo_out.mmax, ainfo_in.mmax)
        self.assertIs(ainfo_out.nelem, ainfo_in.nelem)

    def test_sharp_alm2map_roundtrip(self):
                
        # Test the wrapper around libsharps alm2map/map2alm.
        lmax = 3
        ainfo = sharp.alm_info(lmax)

        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = sharp.map_info_gauss_legendre(nrings, nphi)

        sht = sharp.sht(minfo, ainfo)

        # Test different input shapes and dtypes.
        # Case 1a: 1d double precision.
        spin = 0
        alm = np.zeros((ainfo.nelem), dtype=np.complex128)
        alm[4] = 1. + 1.j

        omap = sht.alm2map(alm, spin=spin)
        self.assertEqual(omap.shape, (minfo.npix,))
        self.assertEqual(omap.dtype, np.float64)
        alm_out = sht.map2alm(omap, spin=spin)

        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 1b: 1d single precision.
        spin = 0
        alm = np.zeros((ainfo.nelem), dtype=np.complex64)
        alm[4] = 1. + 1.j

        omap = sht.alm2map(alm, spin=spin)
        self.assertEqual(omap.shape, (minfo.npix,))
        self.assertEqual(omap.dtype, np.float32)
        alm_out = sht.map2alm(omap, spin=spin)

        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 2a: 2d double precision.
        spin = 1
        nspin = 2
        alm = np.zeros((nspin, ainfo.nelem), dtype=np.complex128)
        alm[0,4] = 1. + 1.j
        alm[1,4] = 2. - 2.j

        omap = sht.alm2map(alm, spin=spin)
        self.assertEqual(omap.shape, (nspin, minfo.npix))
        self.assertEqual(omap.dtype, np.float64)
        alm_out = sht.map2alm(omap, spin=spin)

        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 2b: 2d single precision.
        spin = 1
        nspin = 2
        alm = np.zeros((nspin, ainfo.nelem), dtype=np.complex64)
        alm[0,4] = 1. + 1.j
        alm[1,4] = 2. - 2.j

        omap = sht.alm2map(alm, spin=spin)
        self.assertEqual(omap.shape, (nspin, minfo.npix))
        self.assertEqual(omap.dtype, np.float32)
        alm_out = sht.map2alm(omap, spin=spin)

        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 3a: 3d double precision.
        spin = 1
        nspin = 2
        ntrans = 3
        alm = np.zeros((ntrans, nspin, ainfo.nelem), dtype=np.complex128)
        alm[0,0,4] = 1. + 1.j
        alm[0,1,4] = 2. - 2.j
        alm[1,0,4] = 3. + 3.j
        alm[1,1,4] = 4. - 4.j
        alm[2,0,4] = 5. + 5.j
        alm[2,1,4] = 6. - 6.j

        omap = sht.alm2map(alm, spin=spin)
        self.assertEqual(omap.shape, (ntrans, nspin, minfo.npix))
        self.assertEqual(omap.dtype, np.float64)
        alm_out = sht.map2alm(omap, spin=spin)

        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 3b: 3d single precision.
        spin = 1
        nspin = 2
        ntrans = 3
        alm = np.zeros((ntrans, nspin, ainfo.nelem), dtype=np.complex64)
        alm[0,0,4] = 1. + 1.j
        alm[0,1,4] = 2. - 2.j
        alm[1,0,4] = 3. + 3.j
        alm[1,1,4] = 4. - 4.j
        alm[2,0,4] = 5. + 5.j
        alm[2,1,4] = 6. - 6.j

        omap = sht.alm2map(alm, spin=spin)
        self.assertEqual(omap.shape, (ntrans, nspin, minfo.npix))
        self.assertEqual(omap.dtype, np.float32)
        alm_out = sht.map2alm(omap, spin=spin)

        np.testing.assert_array_almost_equal(alm_out, alm)

    def test_sharp_alm2map_der1(self):
                
        # Test the wrapper around libsharps alm2map_der1.
        lmax = 3
        ainfo = sharp.alm_info(lmax)

        nrings = lmax + 1
        nphi = 2 * lmax + 1
        minfo = sharp.map_info_gauss_legendre(nrings, nphi)

        sht = sharp.sht(minfo, ainfo)

        # Test different input shapes and dtypes.
        # Case 1a: 1d double precision.
        alm = np.zeros((ainfo.nelem), dtype=np.complex128)
        alm[4] = 1. + 1.j

        omap = sht.alm2map_der1(alm)
        # Compare to expected value by doing spin 1 transform
        # on sqrt(ell (ell + 1)) alm.
        alm_spin = np.zeros((2, ainfo.nelem), dtype=np.complex128)
        alm_spin[0] = alm * np.sqrt(2)
        omap_exp = sht.alm2map(alm_spin, spin=1)

        np.testing.assert_array_almost_equal(omap, omap_exp)

        # Case 1b: 1d single precision.
        alm = np.zeros((ainfo.nelem), dtype=np.complex64)
        alm[4] = 1. + 1.j

        omap = sht.alm2map_der1(alm)
        # Compare to expected value by doing spin 1 transform
        # on sqrt(ell (ell + 1)) alm.
        alm_spin = np.zeros((2, ainfo.nelem), dtype=np.complex64)
        alm_spin[0] = alm * np.sqrt(2)
        omap_exp = sht.alm2map(alm_spin, spin=1)

        np.testing.assert_array_almost_equal(omap, omap_exp)

        # Case 2a: 2d double precision.
        ntrans = 3
        alm = np.zeros((ntrans, ainfo.nelem), dtype=np.complex128)
        alm[0,4] = 1. + 1.j
        alm[1,4] = 2. + 2.j
        alm[2,4] = 3. + 3.j

        omap = sht.alm2map_der1(alm)
        # Compare to expected value by doing spin 1 transform
        # on sqrt(ell (ell + 1)) alm.
        alm_spin = np.zeros((ntrans, 2, ainfo.nelem), dtype=np.complex128)
        alm_spin[0,0] = alm[0] * np.sqrt(2)
        alm_spin[1,0] = alm[1] * np.sqrt(2)
        alm_spin[2,0] = alm[2] * np.sqrt(2)
        omap_exp = sht.alm2map(alm_spin, spin=1)

        np.testing.assert_array_almost_equal(omap, omap_exp)

        # Case 2b: 2d single precision.
        ntrans = 3
        alm = np.zeros((ntrans, ainfo.nelem), dtype=np.complex64)
        alm[0,4] = 1. + 1.j
        alm[1,4] = 2. + 2.j
        alm[2,4] = 3. + 3.j

        omap = sht.alm2map_der1(alm)
        # Compare to expected value by doing spin 1 transform
        # on sqrt(ell (ell + 1)) alm.
        alm_spin = np.zeros((ntrans, 2, ainfo.nelem), dtype=np.complex64)
        alm_spin[0,0] = alm[0] * np.sqrt(2)
        alm_spin[1,0] = alm[1] * np.sqrt(2)
        alm_spin[2,0] = alm[2] * np.sqrt(2)
        omap_exp = sht.alm2map(alm_spin, spin=1)

        np.testing.assert_array_almost_equal(omap, omap_exp)

    def test_thumbnails(self):
        print("Testing thumbnails...")

        # Make a geometry far away from the equator
        dec_min = 70 * u.degree
        dec_max = 80 * u.degree
        res = 0.5 * u.arcmin
        shape,wcs = enmap.band_geometry((dec_min,dec_max),res=res)

        # Create a set of point source positions separated by
        # 2 degrees but with 1 column wrapping around the RA
        # direction
        width = 120 * u.arcmin
        Ny = int((dec_max-dec_min)/(width))
        Nx = int((2*np.pi/(width)))
        pys = np.linspace(0,shape[0],Ny)[1:-1]
        pxs = np.linspace(0,shape[1],Nx)[:-1]
        Ny = len(pys)
        Nx = len(pxs)
        xx,yy = np.meshgrid(pxs,pys)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        ps = np.vstack((yy,xx))
        decs,ras = enmap.pix2sky(shape,wcs,ps)
        
        # Simulate these sources with unit peak value and 2.5 arcmin FWHM
        N = ps.shape[1]
        srcs = np.zeros((N,3))
        srcs[:,0] = decs
        srcs[:,1] = ras
        srcs[:,2] = ras*0 + 1
        sigma = 2.5 * u.fwhm * u.arcmin
        omap = pointsrcs.sim_srcs(shape,wcs,srcs,beam=sigma)

        # Reproject thumbnails centered on the sources
        # with gnomonic/tangent projection
        proj = "tan"
        r = 10*u.arcmin
        ret = reproject.thumbnails(omap, srcs[:,:2], r=r, res=res, proj=proj, 
            apod=2*u.arcmin, order=3, oversample=2,pixwin=False)

        # Create a reference source at the equator to compare this against
        ishape,iwcs = enmap.geometry(shape=ret.shape,res=res,pos=(0,0),proj=proj)
        imodrmap = enmap.modrmap(ishape,iwcs)
        model = np.exp(-imodrmap**2./2./sigma**2.)

        # Make sure all thumbnails agree with the reference at the
        # sub-percent level
        for i in range(ret.shape[0]):
            diff = ret[i] - model
            assert np.all(np.isclose(diff,0,atol=1e-3))

    def test_tilemap(self):
        shape, wcs = enmap.fullsky_geometry(30*utils.degree)
        assert shape == (7,12)
        geo  = tilemap.geometry((3,)+shape, wcs, tile_shape=(2,2))
        assert len(geo.active) == 0
        assert np.all(geo.lookup<0)
        assert geo.ntile   == 24
        assert geo.nactive == 0
        assert geo.tile_shape == (2,2)
        assert geo.grid_shape == (4,6)
        assert tuple(geo.tile_shapes[ 0]) == (2,2)
        assert tuple(geo.tile_shapes[ 5]) == (2,2)
        assert tuple(geo.tile_shapes[18]) == (1,2)
        assert tuple(geo.tile_shapes[23]) == (1,2)
        assert geo.ind2grid(7) == (1,1)
        assert geo.grid2ind(1,1) == 7
        geo = geo.copy(active=[1])
        assert geo.nactive == 1
        assert np.sum(geo.lookup>=0) == 1
        assert geo.active[0] == 1
        assert geo.lookup[1] == 0
        geo2 = geo.copy(active=[0,1,2])
        assert geo.nactive == 1
        assert geo2.nactive == 3
        assert geo.compatible(geo) == 2
        assert geo.compatible(geo2) == 1
        geo3 = tilemap.geometry((3,)+shape, wcs, tile_shape=(2,3))
        assert geo.compatible(geo3) == 0
        del geo2, geo3
        m1  = tilemap.zeros(geo.copy(active=[1,2]))
        m2  = tilemap.zeros(geo.copy(active=[2,3,4]))
        m3  = tilemap.zeros(geo.copy(active=[2]))
        for a, i in enumerate(m1.geometry.active): m1.active_tiles[a] = i
        for a, i in enumerate(m2.geometry.active): m2.active_tiles[a] = i*10
        for a, i in enumerate(m3.geometry.active): m3.active_tiles[a] = i*100
        assert m1[0,0] == 1
        assert np.all(m1.tiles[1] == m1.active_tiles[0])
        m12 = m1+m2
        m21 = m2+m1
        assert(m12.nactive == 4)
        assert(m21.nactive == 4)
        assert(np.all(m12.tiles[1] == 1))
        assert(np.all(m21.tiles[1] == 1))
        assert(np.all(m12.tiles[2] == 22))
        assert(np.all(m21.tiles[2] == 22))
        assert(sorted(m12.geometry.active)==sorted(m21.geometry.active))
        m1 += m3
        assert np.all(m1.tiles[2] == 202)
        with self.assertRaises(ValueError): m3 += m1
        m1[:] = 0
        m1c   = np.cos(m1)
        assert m1c.geometry.nactive == 2
        assert np.allclose(m1c, 1)


if __name__ == '__main__':
    unittest.main()
    test_sim_slice()
