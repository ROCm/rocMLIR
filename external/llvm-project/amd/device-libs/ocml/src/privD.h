/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define MATH_MAD(A,B,C) BUILTIN_FMA_F64(A, B, C)

#define MATH_FAST_RCP(X) ({ \
    double _frcp_x = X; \
    double _frcp_ret; \
    _frcp_ret = BUILTIN_AMDGPU_RCP_F64(_frcp_x); \
    _frcp_ret = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-_frcp_x, _frcp_ret, 1.0), _frcp_ret, _frcp_ret); \
    _frcp_ret = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-_frcp_x, _frcp_ret, 1.0), _frcp_ret, _frcp_ret); \
    _frcp_ret; \
})
#define MATH_RCP(X) BUILTIN_DIV_F64(1.0, X)

#define MATH_FAST_DIV(X, Y) ({ \
    double _fdiv_x = X; \
    double _fdiv_y = Y; \
    double _fdiv_ret; \
    double _fdiv_r = BUILTIN_AMDGPU_RCP_F64(_fdiv_y); \
    _fdiv_r = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-_fdiv_y, _fdiv_r, 1.0), _fdiv_r, _fdiv_r); \
    _fdiv_r = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-_fdiv_y, _fdiv_r, 1.0), _fdiv_r, _fdiv_r); \
    _fdiv_ret = _fdiv_x * _fdiv_r; \
    _fdiv_ret = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-_fdiv_y, _fdiv_ret, _fdiv_x), _fdiv_r, _fdiv_ret); \
    _fdiv_ret; \
})
#define MATH_DIV(X,Y) BUILTIN_DIV_F64(X, Y)

#define MATH_FAST_SQRT(X) ({ \
    double _fsqrt_x = X; \
    double _fsqrt_y = BUILTIN_AMDGPU_RSQRT_F64(_fsqrt_x); \
    double _fsqrt_s0 = _fsqrt_x * _fsqrt_y; \
    double _fsqrt_h0 = 0.5 * _fsqrt_y; \
    double _fsqrt_r0 = BUILTIN_FMA_F64(-_fsqrt_h0, _fsqrt_s0, 0.5); \
    double _fsqrt_h1 = BUILTIN_FMA_F64(_fsqrt_h0, _fsqrt_r0, _fsqrt_h0); \
    double _fsqrt_s1 = BUILTIN_FMA_F64(_fsqrt_s0, _fsqrt_r0, _fsqrt_s0); \
    double _fsqrt_d0 = BUILTIN_FMA_F64(-_fsqrt_s1, _fsqrt_s1, _fsqrt_x); \
    double _fsqrt_ret = BUILTIN_FMA_F64(_fsqrt_d0, _fsqrt_h1, _fsqrt_s1); \
    _fsqrt_ret = _fsqrt_x == 0.0 ? _fsqrt_x : _fsqrt_ret; \
    _fsqrt_ret; \
})

#define MATH_SQRT(X) ({ \
    double _sqrt_x = X; \
    bool _sqrt_b = _sqrt_x < 0x1.0p-767; \
    _sqrt_x *= _sqrt_b ? 0x1.0p+256 : 1.0; \
    double _sqrt_y = BUILTIN_AMDGPU_RSQRT_F64(_sqrt_x); \
    double _sqrt_s0 = _sqrt_x * _sqrt_y; \
    double _sqrt_h0 = 0.5 * _sqrt_y; \
    double _sqrt_r0 = BUILTIN_FMA_F64(-_sqrt_h0, _sqrt_s0, 0.5); \
    double _sqrt_h1 = BUILTIN_FMA_F64(_sqrt_h0, _sqrt_r0, _sqrt_h0); \
    double _sqrt_s1 = BUILTIN_FMA_F64(_sqrt_s0, _sqrt_r0, _sqrt_s0); \
    double _sqrt_d0 = BUILTIN_FMA_F64(-_sqrt_s1, _sqrt_s1, _sqrt_x); \
    double _sqrt_s2 = BUILTIN_FMA_F64(_sqrt_d0, _sqrt_h1, _sqrt_s1); \
    double _sqrt_d1 = BUILTIN_FMA_F64(-_sqrt_s2, _sqrt_s2, _sqrt_x); \
    double _sqrt_ret = BUILTIN_FMA_F64(_sqrt_d1, _sqrt_h1, _sqrt_s2); \
    _sqrt_ret *= _sqrt_b ? 0x1.0p-128 : 1.0; \
    _sqrt_ret = ((_sqrt_x == 0.0) | (_sqrt_x == (double)INFINITY)) ? _sqrt_x : _sqrt_ret; \
    _sqrt_ret; \
})

