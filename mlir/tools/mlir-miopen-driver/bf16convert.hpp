#ifndef BFLOAT16_CONVERT_HPP
#define BFLOAT16_CONVERT_HPP

#ifdef __cplusplus
extern "C" {
#endif
typedef unsigned short ushort2  __attribute__((ext_vector_type(2)));
typedef union cvt_bf16_fp32
{
    uint u32;
    ushort2 ushortx2;
    unsigned short ushortvec[2];
    float f32;
} cvt_bf16_fp32_t;

float bfloat16_to_float(ushort src_val)
{
    cvt_bf16_fp32_t target_val;
    target_val.ushortx2 = (ushort2)(0, src_val);
    return target_val.f32;
}

ushort float_to_bfloat16(float src_val)
{
    cvt_bf16_fp32_t target_val;
    target_val.f32 = src_val;
    if((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        if((target_val.u32 & 0xffff) != 0)
        {
            target_val.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    else
    {
        target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));// Round to nearest, round to even
    }
    return target_val.ushortvec[1];
}

#ifdef __cplusplus
}
#endif

#endif // BFLOAT16_CONVERT_HPP
