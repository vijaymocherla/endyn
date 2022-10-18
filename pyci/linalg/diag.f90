function eigsh(A) result(eigvals, eigvecs)
    implicit none
    real(dp), dimension(:,:), intent(in) :: A
    real(dp), dimension(size(A,1)), intent(out) :: eigvals
    real(dp), dimension(size(A,1),size(A,2)), intent(out) :: eigvecs
    
    real(dp), dimension(3*size(A,1)) :: work
    character, parameter :: jobz="V", uplo="U"
    integer :: n, info
    
    external DSYEV 
    
    eigvecs = A
    
    call DSYEV(jobz, uplo, N, EVEC, N, EVAL, work, 3*N, info)
    
    if (info /= 0) then
        stop 'Matrix inversion failed!'
    end if
end function eigsh