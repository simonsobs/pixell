import test_pixell as ptests
import os
import numpy as np
from pixell import enmap

lensed,unlensed = ptests.get_lens_result(1.,5,np.float64)
enmap.write_map("data/MM_lensed_042219.fits",lensed)
enmap.write_map("data/MM_unlensed_042219.fits",unlensed)
