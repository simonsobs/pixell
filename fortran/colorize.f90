subroutine remap(a, res, vals, cols)
	use iso_c_binding
	implicit none
	real(8),    intent(in)    :: a(:), vals(:)
	integer(2), intent(in)    :: cols(:,:)
	integer(2), intent(inout) :: res(:,:)
	real(8) :: v, x(2), y(size(cols,1),2)
	integer :: i, j
	!$omp parallel do private(i,v,x,y)
	do i = 1, size(a)
		v = a(i)
		if(v .ne. v) then
			res(:,i) = 0
			cycle
		end if
		! Find location first greater value in vals
		do j = 1, size(vals)
			if(vals(j) > v) exit
		end do
		! Handle edge cases
		if(j <= 1) then
			res(:,i) = cols(:,j)
		elseif(j > size(vals)) then
			res(:,i) = cols(:,size(vals))
		else
			x = vals(j-1:j)
			y = cols(:,j-1:j)
			res(:,i) = min(max(0,nint(y(:,1) + (v-x(1))*(y(:,2)-y(:,1))/(x(2)-x(1)))),255)
		end if
	end do
end subroutine

subroutine direct(a, res)
	use iso_c_binding
	implicit none
	real(8),    intent(in)    :: a(:,:)
	integer(2), intent(inout) :: res(:,:)
	real(8) :: v
	integer :: i
	!$omp parallel do private(i,v)
	do i = 1, size(a,1)
		v = a(i,1)
		if(v .ne. v) then
			res(:,i) = 0
		else
			res(1:size(a,2),i)  = min(max(0,nint(a(i,:)*256)),255)
			res(size(a,2)+1:,i) = 255
		end if
	end do
end subroutine

subroutine direct_colorcap(a, res)
	! Like direct, but caps jointly across colors instead of
	! individually. This preserves colors of saturated regions
	! instead of just ending up with white. The cost of this is
	! that high noise in a single component leaks into all components
	use iso_c_binding
	implicit none
	real(8),    intent(in)    :: a(:,:)
	integer(2), intent(inout) :: res(:,:)
	real(8) :: v, hue(3), amp, color(3)
	integer :: i, nc
	nc = size(a,2)
	!$omp parallel do private(i,v)
	do i = 1, size(a,1)
		v = a(i,1)+a(i,2)+a(i,3)
		if(v .ne. v) then
			res(:,i) = 0
		else
			! Split rgb into hue and light strength
			hue  = a(i,1:3)
			amp  = max(sum(hue**2)**0.5d0,1d-200)
			hue  = hue/amp
			! Cap the light strength
			amp  = max(0d0,min(1d0,amp))
			! Reconstruct color
			color = hue*amp
			! Then turn this into rgb
			res(1:3,i)  = min(max(0,nint(color*256)),255)
			! Alpha channel handled separately. We copy it from a if available,
			! otherwise set it to 255
			res(4:nc,i) = min(max(0,nint(a(i,4:nc)*256)),255)
			res(nc+1:4,i) = 255
		end if
	end do
end subroutine
