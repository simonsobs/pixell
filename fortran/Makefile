all: interpol_32.f90 interpol_64.f90 array_ops_32.f90 array_ops_64.f90

#%.so: %.f90
#	$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c -m $(basename $<) $< $(LAPACK_LINK) $(OMP_LINK) skip: map_border calc_weights

%_32.f90: %.F90
	perl -pe 's/\b_\b/4/g' < $< > $@
%_64.f90: %.F90
	perl -pe 's/\b_\b/8/g' < $< > $@

clean:
	rm -rf *.pyc *.mod interpol*.f90 array_ops*.f90
