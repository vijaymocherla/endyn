! splitoperator routine 
! 
module sof90_module
implicit none
public run_propagator, eigsh, zmul_mv, zmul_mm, zmul_zdotc, calc_expt
contains
! wrapper for ZGEMV
    subroutine zmul_mv(ndim, A, X, Y)
        integer, intent(in) :: ndim
        double complex, dimension(1:ndim, 1:ndim), intent(in) :: A
        double complex, dimension(1:ndim), intent(in) :: X
        double complex, dimension(1:ndim), intent(out) :: Y
        integer :: m, n, incx, incy, lda
        character :: transa
        double complex :: alpha, beta
        lda = size(A,1)
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
        double complex, dimension(1:ndim, 1:ndim), intent(in) :: A, B
        double complex, dimension(1:ndim, 1:ndim), intent(out) :: C
        integer :: l, m, n, lda, ldb, ldc
        character :: transa, transb, transc
        double complex :: alpha, beta 
        alpha=(1.0d0, 0.0d0)
        beta=(1.0d0, 0.0d0)
        transa = 'N'
        transb = 'N'
        transc = 'N'
        lda = size(A,1)
        ldb = size(B,1)
        ldc = size(C,1)
        l = ndim
        n = ndim
        m = ndim
        call zgemm(transa, transb, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)
    end subroutine    

    ! wrapper for DGEMM
    subroutine dmul_mm(ndim, A, B, C)
        integer, intent(in) :: ndim
        double precision, dimension(1:ndim, 1:ndim), intent(in) :: A, B
        double precision, dimension(1:ndim, 1:ndim), intent(out) :: C
        integer :: l, m, n, lda, ldb, ldc
        character :: transa, transb, transc
        double precision :: alpha, beta 
        alpha = 1.0d0
        beta = 1.0d0
        transa = 'N'
        transb = 'N'
        transc = 'N'
        lda = size(A,1)
        ldb = size(B,1)
        ldc = size(C,1)
        l = ndim
        n = ndim
        m = ndim
        call dgemm(transa, transb, l, n, m, alpha, a, lda, b, ldb, beta, c, ldc)
    end subroutine

    ! wrapper for ZDOTC
    subroutine zmul_zdotc(ndim, X, Y, res)
        integer, intent(in) :: ndim
        double complex, dimension(1:ndim), intent(in) :: X, Y 
        double complex, intent(out) :: res
        double complex, external :: zdotc
        integer :: incx, incy
        incx = 1
        incy = 1
        res = zdotc(ndim, x, incx, y, incy)
    end subroutine    

    ! wrapper for dsyev
    subroutine eigsh(ndim, A, vals, vecs)
    integer, intent(in) :: ndim
    double precision, dimension(1:ndim, 1:ndim), intent(in) :: A
    double precision, dimension(1:ndim), intent(out) :: vals
    double precision, dimension(1:ndim, 1:ndim), intent(out) :: vecs
    integer :: info 
    double precision :: work(3*ndim)
    character, parameter :: jobz="V", uplo="U"
    vecs = A
    call dsyev(jobz, uplo, ndim, vecs, ndim, vals, work, 3*ndim, info)
    end subroutine eigsh

    ! Returns expm(1j*alpha*x)
    subroutine exp_array(ndim, alpha, x, exp_x)
        integer, intent(in) :: ndim
        double precision, intent(in) :: alpha
        double complex, dimension(1:ndim), intent(in) :: x
        double complex, dimension(1:ndim, 1:ndim), intent(out) :: exp_x
        integer :: i 
        double complex :: tmp
        do i = 1, ndim
            tmp = dcmplx(0.0d0, alpha)*x(i)
            exp_x(i,i) = exp(tmp) 
        end do
    end subroutine

    ! subroutine for calculation operator expectations    double precision :: tmp 
    subroutine calc_expt(ndim, nops, yi, ops_list, y0, norm, autocorr, ops_expt)
        integer, intent(in) :: ndim, nops
        double complex, dimension(1:ndim), intent(in) :: yi, y0
        double complex, dimension(1:nops, 1:ndim, 1:ndim), intent(in) :: ops_list
        double precision, dimension(1:nops) :: ops_expt
        double precision, intent(out) :: norm, autocorr
        double complex :: res_norm, res_autocorr, res_exp
        double complex, dimension(1:ndim) :: ytmp
        integer :: i
        call zmul_zdotc(ndim, yi, yi, res_norm)
        norm = abs(res_norm)
        call zmul_zdotc(ndim, yi, y0, res_autocorr)
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
        double precision, parameter :: fs_to_au=41.341374575751
        integer, intent(in) :: ndim, nsteps, print_nstep
        integer, dimension(1:3), intent(in) :: field_params
        double precision, dimension(1:3), intent(in) :: time_params
        double precision, dimension(1:nsteps), intent(in) :: Ex, Ey, Ez
        double complex, dimension(1:ndim, 1:ndim), intent(in) :: eigvecs, HCISD, dpx, dpy, dpz
        double complex, dimension(1:ndim), intent(in) :: eigvals, y0 
        character(*), intent(in) :: outfile
        integer :: nops, tstep_val
        double precision :: t0, tf, dt, ti, ti_fs, norm, autocorr
        double complex, dimension(1:ndim) :: yi_csf, yi_eig
        double complex, dimension(1:4, 1:ndim, 1:ndim) :: ops_list
        double precision, dimension(1:4) :: ops_expt
        double complex, dimension(1:ndim) :: vals_dx, vals_dy, vals_dz
        double complex, dimension(1:ndim, 1:ndim) :: vecs_dx, vecs_dy, vecs_dz
        double complex, dimension(1:ndim, 1:ndim) :: u_dx, u_dy, u_dz, ut_dx, ut_dy, ut_dz
        double complex, dimension(1:ndim, 1:ndim) :: fx, fy, fz
        double precision, dimension(1:ndim) :: dtmp_arr
        double precision, dimension(1:ndim, 1:ndim) :: dtmp_mat
        double complex, dimension(1:ndim) :: ztmp_arr
        double complex, dimension(1:ndim, 1:ndim) :: ztmp_mat
        integer :: k
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
        ! Diagonalizing dipole matrices and building unitary transforms 
        ! between following basis : CI eigen -> CSF -> dipole eigen.
        if (field_params(1) == 1) then
            call eigsh(ndim, real(dpx), dtmp_arr, dtmp_mat)
            vals_dx = dcmplx(dtmp_arr, 0.0d0)
            vecs_dx = dcmplx(dtmp_mat, 0.0d0)
            call zmul_mm(ndim, transpose(vecs_dx), eigvecs, u_dx)
            call zmul_mm(ndim, transpose(eigvecs), vecs_dx, ut_dx)
        end if
        if (field_params(2) == 1) then
            call eigsh(ndim, real(dpy), dtmp_arr, dtmp_mat)
            vals_dy = dcmplx(dtmp_arr, 0.0d0)
            vecs_dy = dcmplx(dtmp_mat, 0.0d0)
            call zmul_mm(ndim, transpose(vecs_dy), eigvecs, u_dy)
            call zmul_mm(ndim, transpose(eigvecs), vecs_dy, ut_dy)
        end if
        if (field_params(3) == 1) then
            call eigsh(ndim, real(dpz), dtmp_arr, dtmp_mat)
            vals_dz = dcmplx(dtmp_arr, 0.0d0)
            vecs_dz = dcmplx(dtmp_mat, 0.0d0)
            call zmul_mm(ndim, transpose(vecs_dz), eigvecs, u_dz)
            call zmul_mm(ndim, transpose(eigvecs), vecs_dz, ut_dz)
        end if
        ! Begining the calculation
        tstep_val = 1
        ! projecting y0_csf into eigen basis
        call  zmul_mv(ndim, transpose(eigvecs), yi_csf, yi_eig)
        open(unit=100, file=outfile)
        ! write headers for outputfile
        write(100, *)
        do while(ti <= tf)
            call calc_expt(ndim, nops, yi_csf, ops_list, yi_csf, norm, autocorr, ops_expt)
            ti_fs = ti * fs_to_au
            write(100, '(7f15.8)') ti_fs, norm, autocorr, &
                            ops_expt(1), ops_expt(2), ops_expt(3), ops_expt(4)
            ! propagating the function
            do k= 1, print_nstep
                ! exp(-1j*dt*H0).yi_eig
                call exp_array(ndim, -dt, eigvals, ztmp_mat)
                call zmul_mv(ndim, ztmp_mat, yi_eig, ztmp_arr)
                yi_eig = ztmp_arr
                if (field_params(1) == 1) then
                    call exp_array(ndim, dt*Ex(k), vals_dx, fx)
                    call zmul_mm(ndim, fx, u_dx, ztmp_mat)
                    call zmul_mm(ndim, ut_dx, ztmp_mat, fx)
                    call zmul_mv(ndim, fx, yi_eig, ztmp_arr)
                    yi_eig = ztmp_arr
                end if
                if (field_params(2) == 1) then
                    call exp_array(ndim, dt*Ey(k), vals_dy, fy)
                    call zmul_mm(ndim, fy, u_dy, ztmp_mat)
                    call zmul_mm(ndim, ut_dy, ztmp_mat, fy)
                    call zmul_mv(ndim, fy, yi_eig, ztmp_arr)
                    yi_eig = ztmp_arr
                end if                 
                if (field_params(3) == 1) then
                    call exp_array(ndim, dt*Ez(k), vals_dz, fz)
                    call zmul_mm(ndim, fz, u_dz, ztmp_mat)
                    call zmul_mm(ndim, ut_dz, ztmp_mat, fz)
                    call zmul_mv(ndim, fz, yi_eig, ztmp_arr)
                    yi_eig = ztmp_arr
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
