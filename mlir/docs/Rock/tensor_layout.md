# Layout with group
<br/>
## tensorflow
### NHWC
<br/>
<br/>
#### When you want to add group in tensorflow,
#### you split on dim of c or k , and tf.concat outputs 
<br/>
##### axis=3 split on c dim ,input layout become [N H W G C]
tf.split api , ,x is the tensor ,axis is which dim you want to split ,split to num_or_size_splits parts
input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
this function will split [N, H, W, C] tensor x along the 3rd dimension, result in groups number of tensors 
with [N, H, W, C/groups] each ,so it's equivalent to [N H W G C]
<br/>
Example of split:
if nhwk is 
2,2,2,2
[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]],[[[[9],[10]],[[11],[12]]],[[[13],[14]],[[15],[16]]]]
=>
<br/>
after split to 2 groups :
2,2,2,1
2,2,2,1
[[[[1]],[[3]]],[[[5]],[[7]]]],[[[[9]],[[11]]],[[[13]],[[15]]]]
[[[[2]],[[4]]],[[[6]],[[8]]]],[[[[10]],[[12]]],[[[14]],[[16]]]]
<br/>
##### user format is [YXGCK], not final format
here we use 4 dim YXCK as user input, but c already /g ,k not split 
weights = tf.get_variable('weights', shape=[filter_height,     
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
<br/>
##### axis=3 split on k dim before transform, after transform 
weight_groups = tf.split(axis=3, num_or_size_splits=groups,value=weights)
YXCK=>YXCGK
<br/>
##### but tensorflow will  transform filter from YXCK to KYXC, weights layout become [GKYXC]:
https://github.com/ROCm/tensorflow-upstream/blob/develop-upstream/tensorflow/core/kernels/conv_ops.cc#L1040-L1048
weight_groups = tf.split(axis=3, num_or_size_splits=groups,value=weights)
YXCK=>YXCGK , after transform => GKYXC
<br/>
after 3 conv done, we merge results of these  input conv weight  ,and merge at the k dim
from: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
results is NHWK
##### finally we concat at dim k ,the layout is [NHWGK]
conv = tf.concat(axis=3, values=output_groups)  
<br/>
<br/>
### NCHW
<br/>
<br/>
match Rock: https://github.com/ROCm/Rock/blob/35142af8a2978d2cb4e3d7854553767943057841/test/cpu_conv.hpp#L85
#### input:[NGCHW]
#### filter:[GKCYX] <= tensorflow transform filter from YXCK to KCYX
#### output:[NGKHW]
<br/>
##pytorch
###NCHW
<br/>
<br/>
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and 
producing half the output channels, and both subsequently concatenated.
#### input:[NGCHW]
#### output:[NGKHW]
https://github.com/pytorch/pytorch/blob/edfc787df494828bcbb2b05b34ad7a316f647b1e/aten/src/ATen/native/ConvolutionMM3d.cpp
auto weight_g = weight.reshape(
        {groups,
         n_output_plane / groups,
         n_input_plane / groups * kernel_depth * kernel_height * kernel_width});
#### filter:[GKCYX]
<br/>
### NHWC
<br/>
<br/>
we can use channels_last to force pytorch use NHWC
b = torch.empty(N, C, H, W, device="cuda:0", memory_format=torch.channels_last)
conv_nhwc = nn.Conv2d(C, K, 3, 1, 1).to("cuda:0", memory_format=torch.channels_last)
we can dump shape like conv_nhwc.shape or a= nn.Conv2d a.weight.shape
then we can judge G in which dim because if G=3 C=6 input=[N 6 H W] ,we can know dim(G) is dim(C)-1
if not , you will get shape like [N 2 3H W] or [3N 2 H W].....
#### input:[NHWGC]
#### filter:[GKCYX]
#### output:[NHWGK]
<br/>
<br/>
if filter:YXCK, tensorflow will transform it and pytorch not use it, we can set GYXCK,
but if it from tensorflow, don't use this format even user set it
from tensorflow: YXCK=> KCYX or KYXC
not from tensorflow : YXCK->YXCK
