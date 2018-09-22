%.c: %.pyx
	cython -o $@ $<
	perl -pi -e 's/typedef npy_float64 _Complex/typedef double _Complex/; s/typedef npy_float32 _Complex/typedef float _Complex/' $@

sharp.c: sharp.pyx csharp.pxd csharp.h

clean:
	rm sharp.c
