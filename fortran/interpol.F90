module fortran

	private :: map_border, calc_weights

contains

pure function sinc(x)
	implicit none
	real(_), intent(in) :: x
	real(_) :: sinc, y
	real(_), parameter :: cut = 1e-4, pi = 3.14159265358979323846d0
	y = abs(pi*x)
	if(y < cut) then
		sinc = 1 - y**2/6 + y**4/120
	else
		sinc = sin(y)/y
	end if
end function

pure function dsinc(x)
	implicit none
	real(_), intent(in) :: x
	real(_) :: dsinc, y
	real(_), parameter :: cut = 1e-4, pi = 3.14159265358979323846d0
	y = abs(pi*x)
	if(y < cut) then
		dsinc = -sin(y)/2
	else
		dsinc = cos(y)/x - sin(y)/(x*y)
	end if
end function

! Port of scipy.ndimage's interpolation, for the purpose of adding transposes.
! It has two main components. One is the so-called spline filter, which I think
! computes relevant B-spline coefficients. It is a 1d operation applied along
! all axes one after another. For a single 1d array a, its action is:
!
! N = sum_i p^i a_i
! a_0 = N
! for i>0: a_i = a_i + p a_{i-1}
! a_{n-1} = p/(p^2-1) * (a_{n-1}+p a_{n-2})
! for i<n-1: a_i = p (a_{i+1}-a_i)
!
! The transpose of an operation of the type 1:n a_i = A a_{i-1} + B a_i
! is n-1::-1 a_i = A a_{i+1} + B a_i. So the transpose of this series of
! operations is:
!
! for i>0: a_i = p (a_{i-1}-a_i)
! a[-2],a[-1] = a[-2]+p/(p^2-1)*p*a[-1], p/(p^2-1)*a[-1]
! for i<n-1: a_i = a_i + p a_{i+1}
! for i>0: a_i = a_i + p^i a_0
!
! p^0 p^1 p^2 p^3 ...
! 0   1   0   0   ...
! 0   0   1   0   ...
! 0   0   0   1   ...
!
! Transpose is
!
! p^0 0   0   0   ...
! p^1 1   0   0   ...
! p^2 0   1   0   ...
! p^3 0   0   1   ...
!
! What about the other case, where max >= len? Here
!  N = (a_0 + p^{n-1} a_{n-1} + sum((p^i+p^{n-1-i})a_i,0<i<n-1))/(1-p^{2*(n-1)})
! This changes the left column of the transposed matrix from p^i to
! c_i, where c_0 = q, c_{n-1} = p^{n-1}*q, c_i = (p^i+p^{n-1-i})*q,
! with q = 1/(1-p^{2*(n-1)}).
!
! Looping through multidimensional D given shape along given axis,
! given a one-dimensional view d. We need 3 loops:
!  1. offsets off noff
!  2. blocks  b   nblock
!  3. samples s   n
! ind = off + (s+b*n)*stride
!
! do off = 0, noff-1
!  do block = 0, nblock-1
!   do samp = 0, nsamp-1
!    i = off + (samp+block*nsamp)*noff

! Apply a 1d spline filter of the given order along
! the given axis of a flat view data of an array
! with shape dims. This is copied from scipy, which
! seems to implement the algorithm described here:
! http://users.fmrib.ox.ac.uk/~jesper/papers/future_readgroups/unser9302.pdf
! It assumes *mirrored* boundary conditions, not cyclic!
subroutine spline_filter1d(data, dims, axis, order, border, trans)
	implicit none
	real(_), intent(inout) :: data(:)
	integer, intent(in)    :: dims(:), axis, order, border
	logical, intent(in)    :: trans
	real(_), allocatable   :: a(:)
	real(_), parameter     :: tolerance = 1d-15
	real(_) :: pole(2), p, weight, pn, p2n, ip, q, v
	integer :: ndim, nblock, noff, n, oi, bi, i, pind, m, npole, pi1, pi2, dpi

	ndim   = size(dims)
	n      = dims(axis+1)
	nblock = product(dims(1:axis))
	noff   = product(dims(axis+2:ndim))

	select case(order)
		case(2); npole = 1; pole(1) = sqrt(8d0)-3
		case(3); npole = 1; pole(1) = sqrt(3d0)-2
		case(4); npole = 2
			pole(1) = sqrt(664d0 - sqrt(438976d0)) + sqrt(304d0) - 19d0
			pole(2) = sqrt(664d0 + sqrt(438976d0)) - sqrt(304d0) - 19d0
		case(5); npole = 2
			pole(1) = sqrt(67.5d0 - sqrt(4436.25d0)) + sqrt(26.25d0) - 6.5d0
			pole(2) = sqrt(67.5d0 + sqrt(4436.25d0)) - sqrt(26.25d0) - 6.5d0
		case default; return
	end select
	weight = product((1-pole(1:npole))*(1-1/pole(1:npole)))
	if(.not. trans) then
		pi1 = 1; pi2 = npole; dpi = 1
	else
		pi1 = npole; pi2 = 1; dpi =-1;
	end if
	!$omp parallel private(oi, bi, a, pind, p, m, pn, i, p2n, ip, q, v)
	allocate(a(n))
	!$omp do collapse(2)
	do oi = 0, noff-1
		do bi = 0, nblock-1
			a = data(oi+bi*n*noff+1:oi+(bi+1)*n*noff:noff)
			a = a * weight
			do pind = pi1, pi2, dpi
				p = pole(pind)
				m = ceiling(log(tolerance)/log(abs(p)))
				! This is a bit cryptic. It was a port of
				! scipy ni_interpolation.c:273. Now it is generalized
				! to support other boundary conditions, as given in
				! https://lorensen.github.io/VTKCodeCoverage/VTK/Imaging/Core/vtkImageBSplineInternals.cxx.gcov.html
				if(.not. trans) then
					! Non-transposed case
					! First initialize the causal filter, which is divided into two cases based on
					! the length of the array. We handle four different initial conditions:
					select case(border)
						case(0) ! Zero boundaries c0+ = a0, so nothing to do
						case(1) ! Nearest neighbor (clamped) initial conditions
							a(1) = a(1)/(1-p)
						case(2) ! Cyclic boundary conditions
							pn = p
							do i = 0, min(n,m)-2
								a(1) = a(1) + pn*a(n-i)
								pn = pn*p
							end do
							if(m >= n) a(1) = a(1)/(1-pn)
						case(3) ! Mirrored boundary conditions
							pn = p
							do i = 2, min(m,n)
								a(1) = a(1) + pn*a(i)
								pn = pn*p
							end do
							if(m >= n) then
								do i = n-1, 2, -1
									a(1) = a(1) + pn*a(i)
									pn = pn*p
								end do
								a(1) = a(1)/(1-pn)
							end if
					end select
					! Perform the forwards recursion
					do i = 2, n
						a(i) = a(i) + p*a(i-1)
					end do
					! Initialize the backwards recursion
					select case(border)
						case(0) ! Zero boundary conditions
							a(n) = -p/(1-p**2)*a(n)
						case(1) ! Nearest neighbor
							a(n) = -p/(1-p**2)/(1-p)*(a(n)-p**2*a(n-1))
						case(2) ! Cyclic
							v = 0
							pn = 1
							do i = 1, min(m,n)
								v = v + a(i)*pn
								pn = pn*p
							end do
							if(m >= n) v = v/(1-pn)
							a(n) = -p*(a(n) + p*v)
						case(3) ! Mirror
							a(n) = -p/(1-p**2)*(a(n)+p*a(n-1))
					end select
					! Perform the backwards recursion
					do i = n-1, 1, -1
						a(i) = p*(a(i+1)-a(i))
					end do
				else
					! Transposed case
					! Perform the transposed backwards recursion
					a(1) = -p*a(1)
					do i = 2, n-1
						a(i) = p*(a(i-1)-a(i))
					end do
					a(n) = a(n) - a(n-1)
					! Initialize the tranposed backwards recursion
					select case(border)
						case(0) ! Zero
							a(n) = -p/(1-p**2)*a(n)
						case(1) ! Nearest
							v      = -p/(1-p**2)/(1-p)
							a(n-1) = a(n-1) - v*p**2 * a(n)
							a(n)   = v * a(n)
						case(2) ! Cyclic
							v = -p**2 * a(n)
							a(n) = -p * a(n)
							if(m >= n) v = v/(1-p**n)
							pn = 1
							do i = 1, min(m,n)
								a(i) = a(i) + v*pn
								pn = pn*p
							end do
						case(3) ! Mirror
							v      = -p/(1-p**2)
							a(n-1) = a(n-1) + v*p * a(n)
							a(n)   = v * a(n)
					end select
					! Perform transposed forwards recursion
					do i = n-1, 1, -1
						a(i) = a(i) + p*a(i+1)
					end do
					! Initialize the tranposed forwards recursion
					select case(border)
						case(0) ! Zero
						case(1) ! Nearest
							a(1) = a(1)/(1-p)
						case(2) ! Cyclic
							if(m >= n) a(1) = a(1)/(1-p**n)
							pn = p
							do i = 0, min(n,m)-2
								a(n-i) = a(n-i) + pn*a(1)
								pn = pn*p
							end do
						case(3) ! Mirror
							if(m >= n) a(1) = a(1)/(1-p**(2*n-2))
							pn = p
							! The order of these two loops doesn't have to
							! be reversed because they commute
							do i = 2, min(m,n)
								a(i) = a(i) + pn*a(1)
								pn = pn*p
							end do
							if(m >= n) then
								do i = n-1, 2, -1
									a(i) = a(i) + pn*a(1)
									pn = pn*p
								end do
							end if
					end select
				end if
			end do
			data(oi+bi*n*noff+1:oi+(bi+1)*n*noff:noff) = a
		end do
	end do
	deallocate(a)
	!$omp end parallel
end subroutine

function get_weight_length(type, order) result(n)
	implicit none
	integer :: type, order, n
	n = 0
	select case(type)
		case(0) ! convolution
			select case(order)
				case(0); n = 1
				case(1); n = 2
				case(3); n = 4
			end select
		case(1) ! spline
			n = order+1
		case(2) ! lanczos
			n = max(1,2*order)
	end select
end function

pure subroutine calc_weights(type, order, p, weights, off)
	implicit none
	integer, intent(in)    :: type, order
	integer, intent(inout) :: off(:)
	real(_), intent(in)    :: p(:)
	real(_), intent(inout) :: weights(:,:)
	integer :: ndim, nw, i, j
	real(_) :: x
	ndim = size(weights, 2)
	nw   = size(weights, 1)
	! Speed up nearest neighbor
	if(order == 0) then
		off = nint(p); weights = 1; return
	end if
	do i = 1, ndim
		off(i) = floor(p(i)-(nw-2)*0.5d0)
		do j = 1, nw
			x = abs(p(i)-(j-1)-off(i))
			weights(j,i) = 0
			select case(type)
				case(0) ! convolution
					select case(order)
						case(0); if(x < 0.5) weights(j,i) = 1
						case(1); if(x < 1.0) weights(j,i) = 1-x
						case(3)
							if    (x < 1) then; weights(j,i) =  1.5*x**3 - 2.5*x**2 + 1
							elseif(x < 2) then; weights(j,i) = -0.5*x**3 + 2.5*x**2 - 4*x + 2; end if
					end select
				case(1) ! spline
					select case(order)
						case(0); if(x < 0.5) weights(j,i) = 1
						case(1); if(x < 1.0) weights(j,i) = 1-x
						case(2)
							if    (x < 0.5) then; weights(j,i) = 0.75-x**2
							elseif(x < 1.5) then; weights(j,i) = 0.50*(1.5-x)**2; end if
						case(3)
							if    (x < 1.0) then; weights(j,i) = (x*x*(x-2)*3+4)/6
							elseif(x < 2.0) then; weights(j,i) = (2-x)**3/6; end if
						case(4)
							if    (x < 0.5) then; weights(j,i) = x**2 * (x**2 * 0.25-0.625)+115d0/192
							elseif(x < 1.5) then; weights(j,i) = x*(x*(x*(5d0/6-x/6)-1.25)+5d0/24)+55d0/96
							elseif(x < 2.5) then; weights(j,i) = (x-2.5)**4/24; end if
						case(5)
							if    (x < 1.0) then; weights(j,i) = x**2*(x**2*(0.25d0-x/12)-0.5d0)+0.55d0
							elseif(x < 2.0) then; weights(j,i) = x*(x*(x*(x*(x/24-0.375d0)+1.25d0)-1.75d0)+0.625d0)+0.425d0
							elseif(x < 3.0) then; weights(j,i) = (3-x)**5/120d0; end if
					end select
				case(2) ! lanczos
					if(order == 0) then
						if(x < 0.5) weights = 1
					else
						if(x < order) weights(j,i) = sinc(x)*sinc(x/order)
					end if
			end select
		end do
	end do
end subroutine

pure subroutine calc_weights_deriv(type, order, p, weights, off)
	implicit none
	integer, intent(in)    :: type, order
	integer, intent(inout) :: off(:)
	real(_), intent(in)    :: p(:)
	real(_), intent(inout) :: weights(:,:)
	integer :: ndim, nw, i, j
	real(_) :: x, sgn
	ndim = size(weights, 2)
	nw   = size(weights, 1)
	! Speed up nearest neighbor
	if(order == 0) then
		off = nint(p); weights = 0; return
	end if
	do i = 1, ndim
		off(i) = floor(p(i)-(nw-2)*0.5d0)
		do j = 1, nw
			x = p(i)-(j-1)-off(i)
			if(x > 0) then; sgn = 1; else; sgn = -1; end if
			x = abs(x)
			weights(j,i) = 0
			select case(type)
				case(0) ! convolution
					select case(order)
						case(0); if(x < 0.5) weights(j,i) = 0
						case(1); if(x < 1.0) weights(j,i) = -1
						case(3)
							if    (x < 1) then; weights(j,i) =  4.5*x**2 - 5*x
							elseif(x < 2) then; weights(j,i) = -1.5*x**2 + 5*x - 4; end if
					end select
				case(1) ! spline
					select case(order)
						case(0); if(x < 0.5) weights(j,i) = 0
						case(1); if(x < 1.0) weights(j,i) = -1
						case(2)
							if    (x < 0.5) then; weights(j,i) = -2*x
							elseif(x < 1.5) then; weights(j,i) = -(1.5-x); end if
						case(3)
							if    (x < 1.0) then; weights(j,i) = x**2 - x*2/3
							elseif(x < 2.0) then; weights(j,i) = -(2-x)**2/2; end if
						case(4)
							if    (x < 0.5) then; weights(j,i) = 2*x * (x**2 * 0.25-0.625d0) + x**2 * 2*x * 0.25
							elseif(x < 1.5) then; weights(j,i) = x**2*5/2 - x**3*2/3  - 2*x*1.25 + 5d0/24
							elseif(x < 2.5) then; weights(j,i) = (x-2.5)**3/6; end if
						case(5)
							! Not computed yet
							!if    (x < 1.0) then; weights(j,i) = x**2*(x**2*(0.25d0-x/12)-0.5d0)+0.55d0
							!elseif(x < 2.0) then; weights(j,i) = x*(x*(x*(x*(x/24-0.375d0)+1.25d0)-1.75d0)+0.625d0)+0.425d0
							!elseif(x < 3.0) then; weights(j,i) = (3-x)**5/120d0; end if
					end select
				case(2) ! lanczos
					if(order == 0) then
						if(x < 0.5) weights = 0
					else
						if(x < order) weights(j,i) = dsinc(x)*sinc(x/order) + sinc(x)*dsinc(x/order)/order
					end if
			end select
			weights(j,i) = sgn*weights(j,i)
		end do
	end do
end subroutine

pure function map_border(border, n, i) result(v)
	implicit none
	integer, intent(in) :: border, n, i
	integer :: v
	if(i < 0 .or. i >= n) then
		select case(border)
			case(0) ! constant value
				v = -1
			case(1) ! nearest
				v = max(0,min(n-1,i))
			case(2) ! cyclic
				v = modulo(i, n)
			case(3) ! mirrored
				v = modulo(i, 2*n-2)
				if(v >= n) v = 2*n-2-v
			case default
				v = -1
		end select
	else
		v = i
	end if
end function

! Interpolates the values of idata at the (0-based) pixel indices given
! by pos, resulting in the array odata. These have the following shapes:
!
!              python         fortran
! idata:   {pre},{ishape}  ngrid,npre
! pos:     ndim,{pdims}    npoint,ndim
! odata:   {pre},{pdims}   npoint,npre
!
! with ngrid = product(ishape) and npoint = product(pdims), and so on.
! type indicates the type of interpolation to use:
!  0: convolution
!  1: spline
!  2: lanczos
! border indicates how to handle borders:
!  0: constant zero
!  1: nearest
!  2: cyclic
!  3: mirror
! trans indicates the transpose operation. In this case the data flow direction
! also changes, so idata will be modified based on odata.
subroutine interpol(idata, ishape, odata, pos, type, order, border, trans)
	implicit none
	real(_), intent(inout) :: idata(:,:), odata(:,:)
	real(_), intent(in)    :: pos(:,:)
	integer, intent(in)    :: ishape(:), type, order, border
	logical, intent(in)    :: trans
	real(_), allocatable   :: weights(:,:)
	real(_) :: v(size(idata,2)), res(size(idata,2))
	integer :: off(size(pos,2)), inds(size(pos,2))
	integer :: xi, si, ci, i, j, dind, ndim, nsamp, nw, npre, ncon, n

	ndim  = size(pos,2)
	nsamp = size(pos,1)
	npre  = size(idata,2)
	nw    = get_weight_length(type, order)
	ncon  = nw**ndim
	if(trans) idata = 0
	!$omp parallel private(si,weights,off,res,inds,ci,dind,i,xi,v,j,n)
	allocate(weights(nw,ndim))
	!$omp do
	do si = 1, nsamp
		call calc_weights(type, order, pos(si,:), weights, off)
		! Multiply each interpolation weight with its corresponding
		! element in idata. For a 2d case with a non-flattened idata D
		! in C order, this would be
		!  D00*W00*W01 D01*W00*W11 D02*W00*W21
		!  D10*W10*W01 D11*W10*W11 D12*W10*W21
		!  D20*W20*W01 D21*W20*W11 D22*W20*W21
		! So loop through each cell of context
		if(.not. trans) res  = 0
		inds = 0
		cloop: do ci = 1, ncon
			! Get the value of this cell of context, taking into
			! account boundary conditions
			dind = 0
			do i = 1,ndim
				xi = inds(i) + off(i)
				n  = ishape(i)
				if(n < 0) write(*,*) "If I don't have this write here, ifort 15 optimizes away ishape and/or n"
				xi = map_border(border, ishape(i), xi)
				! If we don't map onto a valid point (because we use null-boundaries),
				! this cell doesn't contribute, so go to the next one
				if(xi < 0) cycle cloop
				dind = dind * n + xi
			end do
			if(.not. trans) then
				! Standard interpolation
				v = idata(dind+1,:)
				! Now multiply this value by all the relevant weights, one
				! for each dimension
				do i = 1, ndim
					v = v * weights(inds(i)+1,i)
				end do
				res = res + v
			else
				! Transposed interpolation
				v = odata(si,:)
				do i = 1, ndim
					v = v * weights(inds(i)+1,i)
				end do
				do j = 1, npre
					!$omp atomic
					idata(dind+1,j) = idata(dind+1,j) + v(j)
				end do
			end if
			! Advance to next cell
			do i = ndim,1,-1
				inds(i) = inds(i) + 1
				if(inds(i) < nw) exit
				inds(i) = 0
			end do
		end do cloop
		if(.not. trans) odata(si,:) = res
	end do
	deallocate(weights)
	!$omp end parallel
end subroutine

! Like interpol, but implements the derivative of odata with respect to pos.
! This has some problems for 0th and 1st order interpolation:
! 0th order: Derivative is computed as 0 everywhere, even though it should be
!            infinite at pixel boundaries
! 1st order: Derivative has nonsensical values at exact integer positions, where
!            the derivative changes discontinuously. It's broken even if the
!            derivative on each side is the same.
subroutine interpol_deriv(idata, ishape, odata, pos, type, order, border, trans)
	implicit none
	real(_), intent(inout) :: idata(:,:), odata(:,:,:) ! (ngrid,npre) and (npoint,npre,ndim)
	real(_), intent(in)    :: pos(:,:)                 ! (npoint,ndim)
	integer, intent(in)    :: ishape(:), type, order, border
	logical, intent(in)    :: trans
	real(_), allocatable   :: weights(:,:), dweights(:,:)
	real(_) :: v(size(idata,2)), res(size(idata,2),size(pos,2)) ! (npre), (npre,ndim)
	integer :: off(size(pos,2)), inds(size(pos,2))
	integer :: xi, si, ci, i, j, dind, ndim, nsamp, nw, npre, ncon, n

	ndim  = size(pos,2)
	nsamp = size(pos,1)
	npre  = size(idata,2)
	nw    = get_weight_length(type, order)
	ncon  = nw**ndim
	if(trans) idata = 0
	!$omp parallel private(si,weights,dweights,off,res,inds,ci,dind,i,xi,v,j,n)
	allocate(weights(nw,ndim),dweights(nw,ndim))
	!$omp do
	do si = 1, nsamp
		call calc_weights(type, order, pos(si,:), weights, off)
		call calc_weights_deriv(type, order, pos(si,:), dweights, off)
		! Multiply each interpolation weight with its corresponding
		! element in idata. For a 2d case with a non-flattened idata D
		! in C order, this would be
		!  D00*W00*W01 D01*W00*W11 D02*W00*W21
		!  D10*W10*W01 D11*W10*W11 D12*W10*W21
		!  D20*W20*W01 D21*W20*W11 D22*W20*W21
		! So loop through each cell of context
		if(.not. trans) res  = 0
		inds = 0
		cloop: do ci = 1, ncon
			! Get the value of this cell of context, taking into
			! account boundary conditions
			dind = 0
			do i = 1,ndim
				xi = inds(i) + off(i)
				n  = ishape(i)
				xi = map_border(border, ishape(i), xi)
				! If we don't map onto a valid point (because we use null-boundaries),
				! this cell doesn't contribute, so go to the next one
				if(xi < 0) cycle cloop
				dind = dind * n + xi
			end do
			if(.not. trans) then
				! Standard interpolation
				v = idata(dind+1,:)
				! Now multiply this value by all the relevant weights, one
				! for each dimension
				do i = 1, ndim
					v = v * weights(inds(i)+1,i)
				end do
				! Deriv of v1 v2 v3 ... = (dv1/v1 + dv2/v2 + ...)*v1*v2*v3 ...
				do j = 1, npre
					if(v(j) == 0) cycle
					do i = 1, ndim
						res(j,i) = res(j,i) + v(j) * dweights(inds(i)+1,i)/weights(inds(i)+1,i)
					end do
				end do
			else
				! Transposed interpolation
				res = odata(si,:,:)
				do i = 1, ndim
					res(i,:) = res(i,:) * weights(inds(i)+1,i)
				end do
				do j = 1, npre
					do i = 1, ndim
						if(weights(j,i) .ne. 0) then
							!$omp atomic
							idata(dind+1,j) = idata(dind+1,j) + res(j,i) * dweights(inds(i)+1,i)/weights(inds(i)+1,i)
						end if
					end do
				end do
			end if
			! Advance to next cell
			do i = ndim,1,-1
				inds(i) = inds(i) + 1
				if(inds(i) < nw) exit
				inds(i) = 0
			end do
		end do cloop
		if(.not. trans) odata(si,:,:) = res
	end do
	deallocate(weights, dweights)
	!$omp end parallel
end subroutine

!!!!!!! Obsolete stuff below !!!!!!!

! pos[ndim,nout] has indices into pre-flattened idata. nout = product(oshape)
! type indicates the type of interpolation to use. 0 is convolution, 1 is
! spline and 2 is lanczos. border indicates how to handle borders. 0 is
! constant zero value, 1 is nearest, 2 is cyclic and 3 is mirrored. trans indicates
! the transpose operation. It only makes sense when interpolate is a linear operation,
! which it is as long as one doesn't use constant boundary values.
subroutine interpol_old(idata, ishape, odata, pos, type, order, border, trans)
	implicit none
	real(_), intent(inout) :: idata(:,:), odata(:,:)
	real(_), intent(in)    :: pos(:,:)
	integer, intent(in)    :: ishape(:), type, order, border
	logical, intent(in)    :: trans
	real(_), allocatable   :: weights(:,:)
	real(_) :: v(size(idata,1)), res(size(idata,1))
	integer :: off(size(pos,2)), inds(size(pos,2))
	integer :: xi, si, ci, i, j, dind, ndim, nsamp, nw, nsub, ncon, n

	ndim  = size(pos,2)
	nsamp = size(pos,1)
	nsub  = size(idata,1)
	nw    = get_weight_length(type, order)
	ncon  = nw**ndim
	if(trans) idata = 0
	!$omp parallel private(si,weights,off,res,inds,ci,dind,i,xi,v,j,n)
	allocate(weights(nw,ndim))
	!$omp do
	do si = 1, nsamp
		call calc_weights(type, order, pos(si,:), weights, off)
		! Multiply each interpolation weight with its corresponding
		! element in idata. For a 2d case with a non-flattened idata D
		! in C order, this would be
		!  D00*W00*W01 D01*W00*W11 D02*W00*W21
		!  D10*W10*W01 D11*W10*W11 D12*W10*W21
		!  D20*W20*W01 D21*W20*W11 D22*W20*W21
		! So loop through each cell of context
		if(.not. trans) res  = 0
		inds = 0
		cloop: do ci = 1, ncon
			! Get the value of this cell of context, taking into
			! account boundary conditions
			dind = 0
			do i = 1,ndim
				xi = inds(i) + off(i)
				n  = ishape(i)
				if(n < 0) write(*,*) "If I don't have this write here, ifort 15 optimizes away ishape and/or n"
				xi = map_border(border, ishape(i), xi)
				! If we don't map onto a valid point (because we use null-boundaries),
				! this cell doesn't contribute, so go to the next one
				if(xi < 0) cycle cloop
				dind = dind * n + xi
			end do
			if(.not. trans) then
				! Standard interpolation
				v = idata(:,dind+1)
				! Now multiply this value by all the relevant weights, one
				! for each dimension
				do i = 1, ndim
					v = v * weights(inds(i)+1,i)
				end do
				res = res + v
			else
				! Transposed interpolation
				v = odata(:,si)
				do i = 1, ndim
					v = v * weights(inds(i)+1,i)
				end do
				do j = 1, nsub
					!$omp atomic
					idata(j,dind+1) = idata(j,dind+1) + v(j)
				end do
			end if
			! Advance to next cell
			do i = ndim,1,-1
				inds(i) = inds(i) + 1
				if(inds(i) < nw) exit
				inds(i) = 0
			end do
		end do cloop
		if(.not. trans) odata(:,si) = res
	end do
	deallocate(weights)
	!$omp end parallel
end subroutine

! If deriv is true, returns the diagonal of the derivative of interpol(idata, pos)
! with respect to pos. This derivative is a diagonal matrix with respect to
! pixel becuase the value in one interpolated pixel only depends on the position
! that pixel reads off, not the positions any other pixels read off. For each
! element it is a vector, corresponding to the derivative in each direction.
! So if b_i = interpol(a,pos_i), then
! (d b_i)_a / d pos_ja = q_ia delta_ij, where q_ia = dinterpol(a,pos_i)_a.
! odata(nsub,ndim,npos), v(nsub), res(nsub,ndim)
subroutine interpol_deriv_old(idata, ishape, odata, pos, type, order, border, trans)
	implicit none
	real(_), intent(inout) :: idata(:,:), odata(:,:,:)
	real(_), intent(in)    :: pos(:,:)
	integer, intent(in)    :: ishape(:), type, order, border
	logical, intent(in)    :: trans
	real(_), allocatable   :: weights(:,:), dweights(:,:)
	real(_) :: v(size(idata,1)), res(size(idata,1),size(pos,2))
	integer :: off(size(pos,2)), inds(size(pos,2))
	integer :: xi, si, ci, i, j, dind, ndim, nsamp, nw, nsub, ncon, n

	ndim  = size(pos,2)
	nsamp = size(pos,1)
	nsub  = size(idata,1)
	nw    = get_weight_length(type, order)
	ncon  = nw**ndim
	if(trans) idata = 0
	!$omp parallel private(si,weights,dweights,off,res,inds,ci,dind,i,xi,v,j,n)
	allocate(weights(nw,ndim),dweights(nw,ndim))
	!$omp do
	do si = 1, nsamp
		call calc_weights(type, order, pos(si,:), weights, off)
		call calc_weights_deriv(type, order, pos(si,:), dweights, off)
		! Multiply each interpolation weight with its corresponding
		! element in idata. For a 2d case with a non-flattened idata D
		! in C order, this would be
		!  D00*W00*W01 D01*W00*W11 D02*W00*W21
		!  D10*W10*W01 D11*W10*W11 D12*W10*W21
		!  D20*W20*W01 D21*W20*W11 D22*W20*W21
		! So loop through each cell of context
		if(.not. trans) res  = 0
		inds = 0
		cloop: do ci = 1, ncon
			! Get the value of this cell of context, taking into
			! account boundary conditions
			dind = 0
			do i = 1,ndim
				xi = inds(i) + off(i)
				n  = ishape(i)
				xi = map_border(border, ishape(i), xi)
				! If we don't map onto a valid point (because we use null-boundaries),
				! this cell doesn't contribute, so go to the next one
				if(xi < 0) cycle cloop
				dind = dind * n + xi
			end do
			if(.not. trans) then
				! Standard interpolation
				v = idata(:,dind+1)
				! Now multiply this value by all the relevant weights, one
				! for each dimension
				do i = 1, ndim
					v = v * weights(inds(i)+1,i)
				end do
				! Deriv of v1 v2 v3 ... = (dv1/v1 + dv2/v2 + ...)*v1*v2*v3 ...
				do i = 1, ndim
					do j = 1, nsub
						if(v(j) .ne. 0) res(j,i) = res(j,i) + v(j) * dweights(j,i)/weights(j,i)
					end do
				end do
			else
				! Transposed interpolation
				res = odata(:,:,si)
				do i = 1, ndim
					res(:,i) = res(:,i) * weights(inds(i)+1,i)
				end do
				do i = 1, ndim
					do j = 1, nsub
						if(weights(j,i) .ne. 0) then
							!$omp atomic
							idata(j,dind+1) = idata(j,dind+1) + res(j,i) * dweights(j,i)/weights(j,i)
						end if
					end do
				end do
			end if
			! Advance to next cell
			do i = ndim,1,-1
				inds(i) = inds(i) + 1
				if(inds(i) < nw) exit
				inds(i) = 0
			end do
		end do cloop
		if(.not. trans) odata(:,:,si) = res
	end do
	deallocate(weights, dweights)
	!$omp end parallel
end subroutine

end module
