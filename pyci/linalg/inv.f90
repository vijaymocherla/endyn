function dmat_inv(A) result(Ainv)
    implicit none
    real(dp), dimension(:,:), intent(in) :: A
    real(dp), dimension(size(A,1),size(A,2)), intent(out) :: Ainv
    
    real(dp), dimension(size(A,1)) :: work   
    integer, dimension(size(A,1)) :: ipiv   
    integer :: n, info

    external DGETRF
    external DGETRI

    Ainv = A
    n = size(A,1)

    call DGETRF(n, n, Ainv, n, ipiv, info)

    if (info /= 0) then
       stop 'Matrix is numerically singular!'
    end if

    call DGETRI(n, Ainv, n, ipiv, work, n, info)

    if (info /= 0) then
       stop 'Matrix inversion failed!'
    end if
end function inv

function zmat_inv(A) result(Ainv)
    implicit none
    complex(dp), dimension(:,:), intent(in) :: A
    complex(dp), dimension(size(A,1),size(A,2)) :: Ainv

    complex(dp), dimension(size(A,1)) :: work  
    integer, dimension(size(A,1)) :: ipiv  
    integer :: n, info

    external ZGETRF
    external ZGETRI
    
    Ainv = A
    n = size(A,1)
    
    call ZGETRF(n, n, Ainv, n, ipiv, info)
    
    if (info /= 0) then
       stop 'Matrix is numerically singular!'
    end if
    
    call ZGETRI(n, Ainv, n, ipiv, work, n, info)
    
    if (info /= 0) then
       stop 'Matrix inversion failed!'
    end if
end function inv