subroutine test(n, A, b, gram_matrix)
  implicit none
  integer i, j
  integer(4), intent(in) :: n
  real(8),    intent(in) :: A(0 : n - 1, 0 : n - 1)
  real(8),    intent(in) :: b
  real(8),    intent(inout):: gram_matrix(0 : n - 1, 0 : n - 1)


  do i = 0 , n - 1
    do j = i + 1, n - 1
      gram_matrix(j, i) = exp(-b * dot_product((A(i, :)- A(j, :)), (A(i, :)- A(j, :))))
      gram_matrix(i, i) = 1
      gram_matrix(i, j) = gram_matrix(j, i)
    end do
  end do

  write(*, "(a)") "** write from fortran"
  write(*, "(i8)")n
  write(*, "(1p3e12.4)")gram_matrix(1, 1), gram_matrix(1, 2), gram_matrix(1, 3)
  write(*, "(1p3e12.4)")gram_matrix(2, 1), gram_matrix(2, 2), gram_matrix(2, 3)
  write(*, "(1p3e12.4)")gram_matrix(3, 1), gram_matrix(3, 2), gram_matrix(3, 3)

end subroutine test
