#include <algorithm>
#include <cstdint>
#include <cstring>

template <uint32_t Wm, uint32_t We>
__attribute__((noinline)) uint8_t constexpr cast_to_f8(float f_x) {
  static_assert(Wm + We == 7, "Wm+We==7");

  const uint32_t mfmt = 23;
  uint32_t x = 0;
  memcpy(&x, &f_x, sizeof(uint32_t));

  uint32_t head = x & 0xFF800000;
  uint32_t mantissa = x & 0x7FFFFF;
  int exponent = (head >> 23) & 0xFF;
  uint32_t sign = head >> 31;
  uint32_t bias = 127;

  uint32_t signed_all_ones =
      (sign << 7) + ((((1 << We) - 1) << Wm) + ((1 << Wm) - 1));

  // Calcualte maximum singed value FLT_MAX, FLT_MIN
  uint32_t signed_max = signed_all_ones;
  // Deal with inf and NaNs
  if (((x & 0x7F800000) == 0x7F800000))
    return 0x80;
  // handle zero and send denormals to 0
  if (exponent == 0)
    return 0;

  /* First need to check if it is normal or denorm as there is a difference of
  implict 1 Then need to adjust the exponent to align with the F8 exponent, in
  the meanwhile, shift The mantissa. Then for stochastic rounding, add rng to
  mantissa and truncate. And for RNE, no need to add rng. Then probably need to
  check whether there is carry and adjust exponent and mantissa again*/

  // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent
  // bits
  const int f8_bias = (1 << (We - 1u)) - 1 + 1;
  const int f8_denormal_act_exponent =
      1 - f8_bias; // actual exponent of f8 denormal
  /* act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
  f8_exponent is the converted f8 exponent with bias encoding
  exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
  the difference needs to be adjusted and mantissa shifted*/
  int act_exponent = exponent - bias;
  int f8_exponent = 0;
  int exponent_diff = 0;
  // fp32/fp16 is normal with implicit 1
  if (act_exponent <= f8_denormal_act_exponent) {
    /* This is the case where fp32/fp16 is normal but it is in f8 denormal
    range. For example fp8 FNUZ mode, denormal exponent is -7, but if the
    fp32/fp16 actual exponent is -7, it is actually larger due to the implict 1,
    Therefore it needs to be adjust to -6 and mantissa shift right by 1.
    So for fp32/fp16, exponent -8 is the cut point to convert to fp8 FNUZ */
    exponent_diff = f8_denormal_act_exponent - act_exponent;
  } else {             // both fp32/fp16 and f8 are in normal range
    exponent_diff = 0; // exponent_diff=0 does not mean there is no difference
                       // for this case,
    // act_exponent could be larger. Just that it does not need shift mantissa
  }
  mantissa += (1u << mfmt); // Add the implicit 1 into mantissa

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
      (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa) & drop_mask;

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
  const int max_exp = (1 << We) - 1;
  if (f8_exponent > max_exp) {
    return signed_max;
  }

  if (f8_exponent == 0 and mantissa == 0)
    return 0;
  mantissa &= (1 << Wm) - 1;
  return (sign << 7) | (f8_exponent << Wm) | mantissa;
}

extern float v;
extern uint8_t e4m3, e5m2;
int main() {
  e4m3 = cast_to_f8<3, 4>(v);
  e5m2 = cast_to_f8<2, 5>(v);
}
