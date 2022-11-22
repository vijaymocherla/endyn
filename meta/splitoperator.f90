! splitoperator routine 
! 
module splitoperator
implicit none
integer, parameter :: dp=kind(0.d0)
private calc_expt
public run_propagator, eigsh, zmul_mv, zmul_mm, zmul_zdotc
contains

! wrapper for ZGEMV
subroutine zmul_mv(ndim, A, X, Y)
    integer, intent(in) :: ndim
    complex(dp), dimension(1:ndim, 1:ndim), intent(in) :: A
    complex(dp), dimension(1:ndim), intent(in) :: X
    complex(dp), dimension(1:ndim), intent(out) :: Y
    integer :: m, n, incx, incy, lda
    character :: transa
    complex(dp) :: alpha, beta
    lda = 1
    incx = 1
    incy = 1
    m = ndim
    n = ndim
    alpha = (1.0d0, 0.0d0)
    beta = (1.0d0, 0.0d0)
    transa = 'N'
    call zgemv(transa, m, n, alpha, A, lda, X, incx, alpha, Y, incy)
end subroutine    

! wrapper for ZGEMM
subroutine zmul_mm(ndim, A, B, C)
    integer, intent(in) :: ndim
    complex(dp), dimension(1:ndim, 1:ndim), intent(in) :: A, B
    complex(dp), dimension(1:ndim, 1:ndim), intent(out) :: C
    integer :: l, m, n, lda, ldb, ldc
    character :: transa, transb, transc
    complex(dp) :: alpha, beta 
    alpha=(1.0d0, 0.0d0)
    beta=(1.0d0, 0.0d0)
    transa = 'N'
    transb = 'N'
    transc = 'N'
    lda = 1
    ldb = 1
    ldc = 1
    l = ndim
    n = ndim
    m = ndim
    call zgemm(transa, transb, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)
end subroutine    

! wrapper for DGEMM
subroutine dmul_mm(ndim, A, B, C)
    integer, intent(in) :: ndim
    real(dp), dimension(1:ndim, 1:ndim), intent(in) :: A, B
    real(dp), dimension(1:ndim, 1:ndim), intent(out) :: C
    integer :: l, m, n, lda, ldb, ldc
    character :: transa, transb, transc
    real(dp) :: alpha, beta 
    alpha = 1.0d0
    beta = 1.0d0
    transa = 'N'
    transb = 'N'
    transc = 'N'
    lda = 1
    ldb = 1
    ldc = 1
    l = ndim
    n = ndim
    m = ndim
    call dgemm(transa, transb, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)
end subroutine

! wrapper for ZDOTC
subroutine zmul_zdotc(ndim, X, Y, res)
    integer, intent(in) :: ndim
    complex(dp), dimension(1:ndim), intent(in) :: X, Y 
    complex(dp), intent(out) :: res
    complex(dp), external :: zdotc
    integer :: incx, incy
    incx = 1
    incy = 1
    res = zdotc(ndim, x, incx, y, incy)
end subroutine    

! wrapper for dsyev
subroutine eigsh(ndim, A, vals, vecs)
  integer, intent(in) :: ndim
  real(dp), dimension(1:ndim, 1:ndim), intent(in) :: A
  real(dp), dimension(1:ndim), intent(out) :: vals
  real(dp), dimension(1:ndim, 1:ndim), intent(out) :: vecs
  integer :: info 
  double precision :: work(3*ndim)
  character, parameter :: jobz="V", uplo="U"
  vecs = A
  call dsyev(jobz, uplo, ndim, vecs, ndim, vals, work, 3*ndim, info)
end subroutine eigsh

! subroutine for exponentiating an array
subroutine exp_array(ndim, alpha, x, exp_x)
    integer, intent(in) :: ndim
    real(dp), dimension(1:ndim), intent(in) :: x
    real(dp), intent(in) :: alpha
    complex(dp), dimension(1:ndim), intent(out) :: exp_x
    integer :: i 
    complex(dp) :: tmp
    do i = 1, ndim
        tmp = cmplx(0.0d0, alpha*x(i))
        exp_x(i) = exp(tmp) 
    end do
end subroutine

! subroutine for calculation operator expectations
subroutine calc_expt(ndim, nops, yi, ops_list, y0, norm, autocorr, ops_expt)
    integer, intent(in) :: ndim, nops
    complex(dp), dimension(1:ndim), intent(in) :: yi, y0
    complex(dp), dimension(1:nops, 1:ndim, 1:ndim), intent(in) :: ops_list
    real(dp), dimension(1:nops) :: ops_expt
    real(dp), intent(out) :: norm, autocorr
    complex(dp) :: res_norm, res_autocorr, res_exp
    complex(dp), dimension(1:ndim) :: ytmp
    integer :: i
    call zmul_zdotc(ndim, yi, yi, res_norm)
    norm = abs(res_norm)
    call zmul_zdotc(ndim, yi, y0, res_norm)
    autocorr = abs(res_autocorr)
    do i = 1, nops
        call zmul_mv(ndim, ops_list(i, :, :), yi, ytmp)
        call zmul_zdotc(ndim, yi, ytmp, res_exp)
        ops_expt(i) = real(res_exp)
    end do
end subroutine

! main SO routine
subroutine run_propagator(ndim, eigvals, eigvecs, y0, &
                          dpx, dpy, dpz, HCISD, &
                          field_params, Ex, Ey, Ez, &
                          time_params, nsteps, print_nstep,&
                          outfile)
    real(dp), parameter :: fs_to_au=41.341374575751
    integer, intent(in) :: ndim, nsteps, print_nstep
    integer, dimension(1:3), intent(in) :: field_params
    real(dp), dimension(1:3), intent(in) :: time_params
    real(dp), dimension(1:nsteps), intent(in) :: Ex, Ey, Ez
    real(dp), dimension(1:ndim, 1:ndim), intent(in) :: eigvecs, HCISD, dpx, dpy, dpz
    complex(dp), dimension(1:ndim), intent(in) :: eigvals, y0 
    character(*), intent(in) :: outfile
    integer :: nops, i, tstep_val
    real(dp) :: t0, tf, dt, ti, ti_fs, norm, autocorr
    complex(dp), dimension(1:ndim) :: yi_csf, yi_eig
    real(dp), dimension(1:4, 1:ndim, 1:ndim) :: ops_list
    real(dp), dimension(1:4) :: ops_expt
    real(dp), dimension(1:ndim) :: vals_dx, vals_dy, vals_dz
    real(dp), dimension(1:ndim, 1:ndim) :: vecs_dx, vecs_dy, vecs_dz
    real(dp), dimension(1:ndim, 1:ndim) :: u_dx, u_dy, u_dz, ut_dx, ut_dy, ut_dz
    complex(dp), dimension(1:ndim, 1:ndim) :: fx, fy, fz
    t0 = time_params(1)
    tf = time_params(2)
    dt = time_params(3)
    ops_list(1, :, :) = dpx
    ops_list(2, :, :) = dpy
    ops_list(3, :, :) = dpz
    ops_list(4, :, :) = HCISD
    nops = 4
    yi_csf = y0
    ti = t0
    if (field_params(1) == 1) then
        call eigh(ndim, dpx, vals_dx, vecs_dz)
        call dmul_mm(ndim, transpose(vecs_dx), eigvecs, u_dx)
        call dmul_mm(ndim, transpose(eigvecs), vecs_dx, ut_dx)
    end if
    if (field_params(2) == 1) then
        call eigh(ndim, dpy, vals_dy, vecs_dy)
        call dmul_mm(ndim, transpose(vecs_dy), eigvecs, u_dy)
        call dmul_mm(ndim, transpose(eigvecs), vecs_dy, ut_dy)
    end if
    if (field_params(3) == 1) then
        call eigh(ndim, dpz, vals_dz, vecs_dz)
        call dmul_mm(ndim, transpose(vecs_dz), eigvecs, u_dz)
        call dmul_mm(ndim, transpose(eigvecs), vecs_dz, ut_dz)
    end if
    ! Begining the calculation
    tstep_val = 1
    open(unit=100, file=outfile)
    write(100, *)! write headers
    do while(ti <= tf)
        call calc_expt(ndim, nops, yi_csf, ops_list, yi_csf, norm, autocorr, ops_expt)
        ti_fs = ti * fs_to_au
        write(100, '(7f15.8)') ti_fs, norm, autocorr, &
                        ops_expt(1), ops_expt(2), ops_expt(3), ops_expt(4)
        do i= 1, print_nstep
            !propagate the function
            if (field_params(1) == 1) then
                call exp_array(ndim, vals_dx)
            end if
            if (field_params(2) == 1) then
            end if                 
            if (field_params(3) == 1) then
            end if                 
            ti = ti + dt
            tstep_val = tstep_val + 1
        end do 
        call calc_expt(ndim, nops, yi_csf, ops_list, yi_csf, norm, autocorr, ops_expt)
        ti_fs = ti * fs_to_au
        write(100, '(7f15.8)') ti_fs, norm, autocorr, &
                                ops_expt(1), ops_expt(2), ops_expt(3), ops_expt(4)
    end do
    close(100)
end subroutine

end module