! Test to ensure TODO message is emitted for tile OpenMP 5.1 Directives when they are nested.

<<<<<<< HEAD
!RUN: not %flang -fopenmp -fopenmp-version=51 %s 2>&1 | FileCheck %s
=======
!RUN: not %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

subroutine loop_transformation_construct
  implicit none
  integer :: I = 10
  integer :: x
  integer :: y(I)

  !$omp do
  !$omp tile
  do i = 1, I
    y(i) = y(i) * 5
  end do
  !$omp end tile
  !$omp end do
end subroutine

!CHECK: not yet implemented: Unhandled loop directive (tile)
