/* ************************************************************************
 * Copyright (C) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * cop- ies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IM- PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNE- CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <type_traits>

template <typename To, typename From>
inline constexpr To bit_cast(From fr) noexcept {
  static_assert(sizeof(To) == sizeof(From));
  return __builtin_bit_cast(To, fr);
}

// NOLINTBEGIN
template <typename T, bool NegativeZeroNan = true, bool Clip = false>
constexpr uint8_t cast_to_f8(T f_x, uint32_t Wm, uint32_t We,
                             bool stoch = false, uint32_t rng = 0) {
  constexpr bool is_float = std::is_same<T, float>::value;
  // half is not supported for now
  constexpr bool is_half = false;
  // static_assert(Wm + We == 7, "Wm+We==7");
  static_assert(is_float or is_half, "Only float can be cast to f8");

  constexpr uint32_t mfmt = (sizeof(T) == 4) ? 23 : 10;
  typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type x;

  if constexpr (sizeof(T) == 4)
    x = bit_cast<uint32_t>(f_x);
  else
    x = bit_cast<uint16_t>(f_x);

  uint32_t head = 0;
  uint32_t mantissa = 0;
  int exponent = 0;
  uint32_t bias = 0;
  uint32_t sign = 0;
  if constexpr (sizeof(T) == 4) {
    head = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head >> 23) & 0xFF;
    sign = head >> 31;
    bias = 127;
  } else {
    head = x & 0xFC00;
    mantissa = x & 0x3FF;
    exponent = (head >> 10) & 0x1F;
    sign = head >> 15;
    bias = 15;
  }

  uint32_t signed_inf = (sign << 7) + (((1 << We) - 1) << Wm);
  uint32_t signed_all_ones =
      (sign << 7) + ((((1 << We) - 1) << Wm) + ((1 << Wm) - 1));

  // Calcualte maximum singed value FLT_MAX, FLT_MIN
  uint32_t signed_max = signed_all_ones;
  if (not NegativeZeroNan)
    signed_max = (Wm == 2) ? (signed_max - 4) : (signed_max - 1);

  // Deal with inf and NaNs
  if (NegativeZeroNan) // For the FNUZ cases, it is simple just return NaNs
  {
    if ((sizeof(T) == 4 and ((x & 0x7F800000) == 0x7F800000)) or
        (sizeof(T) == 2 and ((x & 0x7C00) == 0x7C00)))
      return 0x80;
  } else {
    // calculate most common NaN mantissa for FP8, which is all Ones in binary
    if ((sizeof(T) == 4 and ((x & 0x7F800000) == 0x7F800000)) or
        (sizeof(T) == 2 and ((x & 0x7C00) == 0x7C00))) {
      // infinity and NaNs
      if (Wm == 3)
        return signed_all_ones;
      return signed_inf | ((mantissa >> (mfmt - Wm)) & ((1 << Wm) - 1));
    }
  }
  // handle positive zero
  if (x == 0)
    return 0;
  // handle negative zero
  else if ((sizeof(T) == 4 and x == 0x80000000) or
           (sizeof(T) == 2 and x == 0x8000)) {
    return NegativeZeroNan
               ? 0
               : 0x80; // For FNUZ types neg zero is just positive zero
  }

  /* First need to check if it is normal or denorm as there is a difference of
  implict 1 Then need to adjust the exponent to align with the F8 exponent, in
  the meanwhile, shift The mantissa. Then for stochastic rounding, add rng to
  mantissa and truncate. And for RNE, no need to add rng. Then probably need to
  check whether there is carry and adjust exponent and mantissa again*/

  // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent
  // bits
  const int f8_bias = (1 << (We - 1u)) - 1 + (NegativeZeroNan ? 1 : 0);
  const int f8_denormal_act_exponent =
      1 - f8_bias; // actual exponent of f8 denormal
  /* act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
  f8_exponent is the converted f8 exponent with bias encoding
  exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
  the difference needs to be adjusted and mantissa shifted*/
  int act_exponent = 0;
  int f8_exponent = 0;
  int exponent_diff = 0;

  if (exponent == 0 and mantissa != 0) { // fp32/fp16 is in denormal.
    /* fp32 denormal is below 2^-127 so it is usually not a concern here, we
    mostly concern fp16 here. In this case, f8 is usually in denormal. But there
    could be exceptions. fp16 denormal has exponent bias 15 while bf8 with FNUZ
    has exponent bias 16. It means that there are some numbers in fp16 denormal
    but they are bf8 (FNUZ) normals - smallest bf8 (FNUZ) normal is 2^-15. fp16
    numbers where exponent==0 (actual exponent -14) and highest bit of mantissa
    is 1 are bf8 (FNUZ) normal. In this case, the fp16 mantissa should be shift
    left by 1  */
    act_exponent = 1 - bias;
    exponent_diff =
        f8_denormal_act_exponent -
        act_exponent; // actual exponent is exponent-bias+1 as it is denormal
  } else {            // fp32/fp16 is normal with implicit 1
    act_exponent = exponent - bias;
    if (act_exponent <= f8_denormal_act_exponent) {
      /* This is the case where fp32/fp16 is normal but it is in f8 denormal
      range. For example fp8 FNUZ mode, denormal exponent is -7, but if the
      fp32/fp16 actual exponent is -7, it is actually larger due to the implict
      1, Therefore it needs to be adjust to -6 and mantissa shift right by 1. So
      for fp32/fp16, exponent -8 is the cut point to convert to fp8 FNUZ */
      exponent_diff = f8_denormal_act_exponent - act_exponent;
    } else {             // both fp32/fp16 and f8 are in normal range
      exponent_diff = 0; // exponent_diff=0 does not mean there is no difference
                         // for this case,
      // act_exponent could be larger. Just that it does not need shift mantissa
    }
    mantissa += (1u << mfmt); // Add the implicit 1 into mantissa
  }

  // need to know whether the number is right in the middle of two adjacent fp8
  // numbers. use  max value of 31 to avoid undefined behaviour
  bool midpoint =
      (mantissa & ((1u << std::min(31u, mfmt - Wm + exponent_diff)) - 1)) ==
      (1u << std::min(31u, mfmt - Wm + exponent_diff - 1));
  /* This part is a bit tricky. The judgment of whether it is a tie needs to be
  done before we shift right as shift right could rip off some residual part and
  make something not midpoint look like midpoint. For example, the fp16 number
  0x1002 (0 00100 0000000010), it is larger than midpoint, but after shift right
  by 4 bits, it would look like midpoint.
  */

  if (exponent_diff > 0)
    mantissa >>= std::min(31u, uint32_t(exponent_diff));
  else if (exponent_diff == -1)
    mantissa <<= -exponent_diff;
  bool implicit_one = mantissa & (1 << mfmt);
  // if there is no implict 1, it  means the f8 is denormal and need to adjust
  // to denorm exponent
  f8_exponent = (act_exponent + exponent_diff) /*actual f8 exponent*/ +
                f8_bias - (implicit_one ? 0 : 1);

  // Now we have the exponent and mantissa adjusted
  uint32_t drop_mask = (1u << (mfmt - Wm)) - 1;
  bool odd =
      mantissa &
      (1u << (mfmt -
              Wm)); // if the least significant bit that is not truncated is 1
  /*
  This part is doing rounding by adding mantissa part that is going to get
  dropped. e.g. if the dropped part for less than 0.5 than it would round down.
  if the dropped part is more than 0.5 then it would round up by rolling carry
  to LSB of retained mantissa. For the mid point when bit pattern is like this
  for Odd: `xy1:10000000` for Odd and `xy0:10000000` for the Even.  where `:` is
  delimiter for dropped v/s retained part. For the odd case : this will add
  xy1:10000000 + 000:10000000 which would roll over carry to LSB of retained
  part making it RNE.
  For the even case : this will add xy0:10000000 + 000:01111111 which would
  round down and keep number Even
  */
  mantissa +=
      (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa)) &
      drop_mask;

  // Now we deal with overflow
  if (f8_exponent == 0 and ((1 << mfmt) & mantissa)) {
    f8_exponent = 1; // denormal overflow to become normal, promote exponent
  } else if ((1 << (mfmt + 1)) & mantissa) {
    mantissa >>= 1;
    f8_exponent++;
  }

  mantissa >>= (mfmt - Wm);

  // above range: quantize to maximum possible float of the same sign
  // for e5m2 case, max_exp is 14, since exp = 15 is reserved for Infs and Nans
  const int max_exp = (1 << We) - ((NegativeZeroNan or Wm == 3) ? 1 : 2);
  if (f8_exponent > max_exp) {
    if (Clip)
      return signed_max;
    else {
      // https://onnx.ai/onnx/technical/float8.html#cast
      if (NegativeZeroNan)
        return 0x80;
      else
        return (Wm == 2) ? signed_inf : signed_all_ones;
    }
  }

  if (f8_exponent == 0 and mantissa == 0)
    return NegativeZeroNan ? 0 : (sign << 7);
  mantissa &= (1 << Wm) - 1;
  return (sign << 7) | (f8_exponent << Wm) | mantissa;
}
// NOLINTEND

extern uint8_t e4m3, e5m2;

void foo(float v) {
  e4m3 = cast_to_f8<float, false, false>(v, 3, 4);
  e5m2 = cast_to_f8<float, false, false>(v, 2, 5);
}
