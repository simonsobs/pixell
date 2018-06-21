include ../compile_opts/$(ENLIB_COMP).mk
all: sharp.so tidy
sharp.so: sharp.c csharp.h
	$(PYTHON) setup.py build_ext --inplace
sharp.c: sharp.pyx csharp.pxd
	cython sharp.pyx
	sed 's/typedef npy_float64 _Complex/typedef double _Complex/;s/typedef npy_float32 _Complex/typedef float _Complex/' sharp.c > tmp.c
	mv tmp.c sharp.c
clean:
	rm -rf *.so *.pyc sharp.c
tidy:
	rm -rf build
