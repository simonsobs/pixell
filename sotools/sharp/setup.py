from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess, sys
extra_link_args = ["-fopenmp"]
#try:
#	extra_link_args += subprocess.check_output(["mpicc", "--showme:link"]).split()
#except (OSError, subprocess.CalledProcessError) as e:
#	print "\033[0;31mCould not find mpi link options!\nSkipping mpi linkage. This will cause problems if libsharp was compiled with mpi support\033[0m"

setup(
	name="sharp",
	cmdclass = {"build_ext": build_ext},
	ext_modules = [
		Extension(
			name="sharp",
			sources=["sharp.c"],
			libraries=["sharp","c_utils","fftpack"],
			include_dirs=["."],
			extra_link_args = extra_link_args,
			)
		]
	)
