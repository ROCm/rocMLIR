module {
    func @main(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>){
      //%0 = alloc() : memref<128x8x30x30xf32>
      //%1 = alloc() : memref<128x8x30x30xf32>
      call @step_1(%arg0, %arg1) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
      call @step_2(%arg0, %arg1) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
      return
    }

    func @step_1(%A : memref<?x?x?x?xf32>, %B : memref<?x?x?x?xf32>) {
    affine.for %i = 0 to 42 {
        affine.for %j = 0 to 10 {
        affine.for %ii = 2 to 16 {
            affine.for %jj = 5 to 17 {
            %0 = load %A[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>
            store %0, %B[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>
            }
        }
        }
    }
    return
    }
    func @step_2(%A : memref<?x?x?x?xf32>, %B : memref<?x?x?x?xf32>) {
    affine.for %i = 0 to 42 {
        affine.for %j = 0 to 10 {
        affine.for %ii = 2 to 16 {
            affine.for %jj = 5 to 17 {
            %0 = load %A[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>
            store %0, %B[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>
            }
        }
        }
    }
    return
    }

}