import sys
sys.path.append('../../tests')
import test_pixell as ptests
import os
import numpy as np
from pixell import enmap

obs_pos,grad,raw_pos = ptests.get_offset_result(1.,np.float64)
enmap.write_map("MM_offset_obs_pos_042219.fits",obs_pos)
enmap.write_map("MM_offset_grad_042219.fits",grad)
enmap.write_map("MM_offset_raw_pos_042219.fits",raw_pos)

lensed,unlensed = ptests.get_lens_result(1.,400,np.float64)
enmap.write_map("MM_lensed_042219.fits",lensed)
enmap.write_map("MM_unlensed_042219.fits",unlensed)
