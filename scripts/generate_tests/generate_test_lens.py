import sys
sys.path.append('../../tests')
import test_pixell as ptests
import os
import numpy as np
from pixell import enmap

version = sys.argv[1]

obs_pos,grad,raw_pos = ptests.get_offset_result(1.,np.float64)
enmap.write_map("MM_offset_obs_pos_%s.fits" % version,obs_pos)
enmap.write_map("MM_offset_grad_%s.fits"  % version,grad)
enmap.write_map("MM_offset_raw_pos_%s.fits"  % version,raw_pos)

lensed,unlensed = ptests.get_lens_result(1.,400,np.float64)
enmap.write_map("MM_lensed_%s.fits"  % version,lensed)
enmap.write_map("MM_unlensed_%s.fits"  % version,unlensed)
