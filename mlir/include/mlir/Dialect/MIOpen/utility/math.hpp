#ifndef MATH_HPP
#define MATH_HPP

namespace math {
// greatest common divisor, aka highest common factor
template <typename T> T gcd(T x, T y) {
  assert(!(x == 0 && y == 0));

  if (x < 0 || y < 0) {
    return gcd(abs(x), abs(y));
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

} // namespace math
#endif