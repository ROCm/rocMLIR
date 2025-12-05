rocMLIR uses im2col to transform convolutions to GEMMs. 
Considering kernel of shape `(K, r, r, C)` and input of shape `(N, H, W, C)` it would generate 

GEMM -> `(N, OH, OW, r * r * C) * (r * r * C, K)`
Here, 
r = kernel_height & kernel_width == kernel_height. 
C = input channels, 
K = Output channels
N = Batch
OW = output width
OH = output height


Convolution handles padding internally as well. 

To implement input transforms inside rocMLIR, We could think of `B^T * d * B` as two convolutions. 
To extract sliding windows of size `(m + r - 1 x m + r -1)` with stride `m` we can first try to do im2col but that would require some amendments. 

im2Col will generate input matrix of shape, 
`{N, tileH * m, tileW * m, (m +r -1) * (m + r -1) * C}`
We don't want to do reduction over `C` therefore before doing im2col first move C axis using transpose

`{N, H, W, C} -> {N * C, H, W, 1}` 

We can mark input channel = 1 in the convolution operation. 

After im2col -> `{N * C, tileH * m, tileW * m, alphaH * alphaW}` where `alphaH = m + r - 1 = alphaW`
Filter for the convolution would be `B^T` which is of shape `{alphaH, alphaW}` we can broadcast and reshape it to be of shape 
`{N,  C, (alphaH * alphaW), 1}` 

After im2col it will do batched gemm 
`{N * C, (tileH *m  * tileW * m), (alphaH * alphaW)} * {N * C, (alphaH * alphaW,) 1}`

But doing GEMM using im2col will reduce entire window to single element.  We don't want that. Instead we want to do matrix multiplication using window from input and window from filter.  Therefore after doing im2col it requires some amendments.

Input shape after im2col 
`{N * C, tileH *m, tileW * m , alpha, alpha}` 
transpose last two axes
`{N * C, tileH * m, tileW *m, alpha, alpha}`
transpose again
`{N * C  * alpha, tileH * m, tileW *m, alpha}`
transpose again and reshape
`{N * C * alpha, alpha, tileH * m * tileW *m}
Filter   : 
`{N, C, (alpha * alpha), 1}`

transform it to be `{N * C * alpha, 1, alpha}` 

Then do GEMM 
`{N * C * alpha, 1, alpha} * {N * C * alpha, alpha, tileH * m * tileW *m}`

Note that it dot product is between `alphaW (from filter)* alphaH(from input)`

Ouptut `{N * C * alpha, 1, tileH * m  * tileW * m }`

