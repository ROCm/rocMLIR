module {
    func @main(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>){
      call @step_1(%arg0, %arg1) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> ()
      return
    }

    func private @step_1(%A : tensor<?x?x?x?xf32>, %B : tensor<?x?x?x?xf32>) {
    %0 = tensor_to_memref %A : memref<?x?x?x?xf32>
    %1 = tensor_to_memref %B : memref<?x?x?x?xf32>

    affine.for %i = 0 to 42 {
        affine.for %j = 0 to 10 {
        affine.for %ii = 2 to 16 {
            affine.for %jj = 5 to 17 {
            %2 = load %0[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>
            store %2, %1[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>
            }
        }
        }
    }
    return
    }
    
}