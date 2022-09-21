#ifndef MATH_HPP
#define MATH_HPP

#include <cassert>
#include <cstdlib>

namespace math_util {
// greatest common divisor, aka highest common factor
template <typename T> T gcd(T x, T y) {
  assert(!(x == 0 && y == 0));

  if (x < 0 || y < 0) {
    return gcd(std::abs(x), std::abs(y));
  } else if (x == y || x == 0) {
    return y;
  } else if (y == 0) {
    return x;
  } else if (x > y) {
    return gcd(x % y, y);
  } else {
    return gcd(x, y % x);
  }
}

template <typename X, typename... Ys> auto gcd(X x, Ys... ys) {
  return gcd(x, ys...);
}

// least common multiple
template <typename T> T lcm(T x, T y) { return (x * y) / gcd(x, y); }

template <typename X, typename... Ys> auto lcm(X x, Ys... ys) {
  return lcm(x, lcm(ys...));
}

template <class X, class Y> auto integer_divide_ceil(X x, Y y) {
  return (x + y - 1) / y;
}

template <class X, class Y> constexpr auto integer_divide_floor(X x, Y y) {
  return x / y;
}

template <class X, class Y> auto integer_least_multiple(X x, Y y) {
  return y * integer_divide_ceil(x, y);
}

/// Returns a % b except that when a % b, returns b
template <class X> constexpr X mod_1_to_n(X a, X b) {
  return (a % b == 0) ? b : a % b;
}
} // namespace math_util
#endif
