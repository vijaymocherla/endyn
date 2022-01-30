program main
    implicit none

end program main

subroutine eigsh(N, MAT, EVAL, EVEC)
    implicit none

    integer, intent(in)           :: N
    double precision, intent(in)  :: MAT(N,N)
    double precision, intent(out) :: EVAL(N)
    double precision, intent(out) :: EVEC(N,N)
  
    integer                       :: info 
    double precision              :: work(3*N)
    character, parameter          :: jobz="V", uplo="U"
  
    integer                       :: i

    EVEC = MAT
    
    call dsyev(jobz, uplo, N, EVEC, N, EVAL, work, 3*N, info)

end subroutine eigsh