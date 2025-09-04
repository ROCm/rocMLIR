! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -pedantic

<<<<<<< HEAD
!PORTABILITY: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg1 has length 64, which is greater than the maximum name length 63 [-Wlong-names]
=======
!PORTABILITY: AAAAAAAAAABBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEEEEEEEEEFFFFFFFFFFGGG1 has length 64, which is greater than the maximum name length 63 [-Wlong-names]
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
program aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg1

  !PORTABILITY: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg2 has length 64, which is greater than the maximum name length 63 [-Wlong-names]
  integer :: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg2

  integer :: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg

  !PORTABILITY: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg3 has length 64, which is greater than the maximum name length 63 [-Wlong-names]
  call aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg3

end
