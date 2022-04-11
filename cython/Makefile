all: sharp.c distances.c srcsim.c

%.c: %.pyx
	cython --fast-fail -o $@ $<
	perl -pi -e 's/typedef npy_float64 _Complex/typedef double _Complex/; s/typedef npy_float32 _Complex/typedef float _Complex/' $@

sharp.c: sharp.pyx csharp.pxd csharp.h sharp_utils.c sharp_utils.h sharp_utils.c

distances.c: distances.pyx distances_wrap.pxd distances_core.h distances_core.c

srcsim.c: srcsim.pyx srcsim_wrap.pxd srcsim_core.h srcsim_core.c

clean:
	rm -f sharp.c distances.c srcsim.c
