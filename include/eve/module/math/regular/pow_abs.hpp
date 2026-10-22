//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Project Contributors
  SPDX-License-Identifier: BSL-1.0
*/
//==================================================================================================
#pragma once

#include <eve/arch.hpp>
#include <eve/traits/overload.hpp>
#include <eve/module/core/decorator/core.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math/constant/maxlog.hpp>
#include <eve/module/math/constant/minlog.hpp>
#include <eve/module/math/constant/minlogdenormal.hpp>
#include <eve/module/math/constant/log_2.hpp>
#include <eve/module/math/regular/reverse_horner.hpp>
#include <eve/module/math/regular/exp.hpp>
#include <eve/module/math/regular/log.hpp>

namespace eve
{
  template<typename Options>
  struct pow_abs_t : elementwise_callable<pow_abs_t, Options, raw_option, fast_option, pedantic_option>
  {
    template<eve::floating_scalar_value T, eve::integral_scalar_value U>
    EVE_FORCEINLINE constexpr T operator()(T v, U w) const noexcept
    { return EVE_DISPATCH_CALL(v, w); }

    template<eve::floating_value T, eve::floating_value U>
    requires(eve::same_lanes_or_scalar<T, U>)
    EVE_FORCEINLINE constexpr common_value_t<T, U> operator()(T v, U w) const noexcept
    { return EVE_DISPATCH_CALL(v, w); }

    EVE_CALLABLE_OBJECT(pow_abs_t, pow_abs_);
  };

//================================================================================================
//! @addtogroup eve_math_exp
//! @{
//! @var pow_abs
//!
//! @brief Callable object computing the pow_abs function \f$|x|^y\f$.
//!
//!   @groupheader{Header file}
//!
//!   @code
//!   #include <eve/module/math.hpp>
//!   @endcode
//!
//!   @groupheader{Callable Signatures}
//!
//!   @code
//!   namespace eve
//!   {
//!      // Regular overload
//!      constexpr auto pow_abs(floating_value auto x, floating_value auto y)                          noexcept; // 1
//!
//!      // Lanes masking
//!      constexpr auto pow_abs[conditional_expr auto c](floating_value auto x, floating_value auto y) noexcept; // 2
//!      constexpr auto pow_abs[logical_value auto m](floating_value auto x, floating_value auto y)    noexcept; // 2
//!
//!      // Semantic options
//!      constexpr auto pow_abs[raw](floating_value auto x, floating_value auto y)                     noexcept; // 3
//!   }
//!   @endcode
//!
//! **Parameters**
//!
//!    * `x`, `y`: [floating_value](@ref eve::floating_value) arguments.
//!    * `c`: [Conditional expression](@ref eve::conditional_expr) masking the operation.
//!    * `m`: [Logical value](@ref eve::logical_value) masking the operation.
//!
//! **Return value**
//!
//! Returns [elementwise](@ref glossary_elementwise) \f$|x|^y\f$.
//!
//!   1. The result type is the [common type](@ref common_value_t) of the two parameters.
//!      In particular we have (IEC 60559) for floating entries:
//!
//!      *  pow_abs(\f$\pm0\f$, y), where y is a negative odd integer, returns \f$+\infty\f$.
//!      *  pow_abs(\f$\pm0\f$, y), where y is negative, finite, and is an even integer or a
//!      non-integer, returns \f$+\infty\f$.
//!      *  pow_abs(\f$\pm0\f$, \f$-\infty\f$) returns \f$+\infty\f$.
//!      *  pow_abs(\f$\pm0\f$, y), where y is a positive odd integer, returns \f$+0\f$.
//!      *  pow_abs(\f$\pm0\f$, y), where y is positive non-integer or a positive even integer, returns
//!      \f$+0\f$.
//!      *  pow_abs(-1, \f$\pm\infty\f$) returns 1.
//!      *  pow_abs(\f$\pm1\f$, y) returns 1 for any y, even when y is NaN.
//!      *  pow_abs(x, \f$\pm0\f$) returns 1 for any x, even when x is NaN.
//!      *  pow_abs(x, \f$-\infty\f$) returns \f$+\infty\f$ for any |x| `<` 1.
//!      *  pow_abs(x, \f$-\infty\f$) returns \f$+0\f$ for any |x| `>` 1.
//!      *  pow_abs(x, \f$+\infty\f$) returns \f$+0\f$ for any |x| `<` 1.
//!      *  pow_abs(x, \f$+\infty\f$) returns \f$+\infty\f$ for any |x| `>` 1.
//!      *  pow_abs(\f$+\infty\f$, y) returns \f$+0\f$ for any negative y.
//!      *  pow_abs(\f$+\infty\f$, y) returns \f$+\infty\f$ for any positive y.
//!      *  except where specified above, if any argument is NaN, NaN is returned.
//!   2. [The operation is performed conditionally](@ref conditional)
//!   3. faster but less accurate call.
//!
//!  @groupheader{Example}
//!  @godbolt{doc/math/pow_abs.cpp}
//================================================================================================
  inline constexpr auto pow_abs = functor<pow_abs_t>;
//================================================================================================
//!  @}
//================================================================================================

  namespace _
  {
    // |x|^y as exp(y*log|x|), with log|x| evaluated as an unevaluated sum lh + ll and y*log|x| as
    // th + tl.  The SIMD pow_abs kernel uses it for the lanes it cannot handle by binary
    // powering.  A finite non-zero result needs |y*log|x|| below about 745 (double) or 104
    // (float).  The error is mostly that of exp: on sampled inputs over that range the result
    // stayed within 1 (double) or 2 (float) representable values of the correctly rounded one,
    // i.e. within about 1.3 or 1.8 ulp of the exact value (measured, not a proven bound).
    template<floating_simd_value T>
    EVE_NOINLINE T pow_abs_large(T ax, T y) noexcept
    {
      // Product and sum of two unevaluated sums hi + lo (lo may be zero), renormalised.
      auto mul2 = [](auto ah, auto al, auto bh, auto bl) {
        auto [p, e] = two_prod(ah, bh);
        e = fma(ah, bl, fma(al, bh, e));
        return two_add(p, e);
      };
      auto add2 = [](auto ah, auto al, auto bh, auto bl) {
        auto [s, e] = two_add(ah, bh);
        return two_add(s, e + al + bl);
      };
      // A lane here has a normal x or |y| > 31 (float) or 63 (double).  When x is not a normal
      // number, |y*log|x|| is then far outside the range of exp, and stays so if x is clamped to
      // the normal range.  Clamping thus takes care of 0, denormals and infinity; a NaN x is
      // restored at the end.
      auto mn = smallestposval(as(ax));
      auto mx = valmax(as(ax));
      ax      = if_else(ax < mn, mn, ax);
      ax      = if_else(ax > mx, mx, ax);
      // ax = m 2^k, m in [0.75, 1.5[, and log(m) = 2 atanh(s), s = (m-1)/(m+1) in [-1/7, 1/5]
      auto [m, k] = frexp(ax);
      auto mlt    = m < T(0.75);
      m           = if_else(mlt, m + m, m);
      k           = dec[mlt](k);
      auto n      = dec(m); // exact
      auto [dh, dl] = two_add(m, one(as(m)));
      auto sh     = n / dh;
      auto sl     = fnma(sh, dl, fnma[pedantic](sh, dh, n)) / dh; // exact remainder
      // 2 atanh(s) = 2s + (2/3) s^3 + s^5 P(s^2), P(u) = sum_j 2u^j/(2j+5) truncated where its
      // terms no longer matter at working precision (s^2 <= 1/25)
      auto [s2h, s2l] = mul2(sh, sl, sh, sl);
      T    p;
      if constexpr( std::same_as<element_type_t<T>, float> )
        p = eve::reverse_horner(s2h, T(2.f/5), T(2.f/7), T(2.f/9), T(2.f/11), T(2.f/13));
      else
        p = eve::reverse_horner(s2h, T(2./5), T(2./7), T(2./9), T(2./11), T(2./13), T(2./15)
                               , T(2./17), T(2./19), T(2./21), T(2./23), T(2./25));
      T const c23h = ieee_constant<0x1.5555555555555p-1, 0x1.555556p-1f>(as<T>{});
      T const c23l = ieee_constant<0x1.5555555555555p-55, -0x1.555556p-26f>(as<T>{});
      T const ln2h = ieee_constant<0x1.62e42fefa39efp-1, 0x1.62e430p-1f>(as<T>{});
      T const ln2l = ieee_constant<0x1.abc9e3b39803fp-56, -0x1.05c610p-29f>(as<T>{});
      auto [s3h, s3l] = mul2(s2h, s2l, sh, sl);
      auto [s5h, s5l] = mul2(s2h, s2l, s3h, s3l);
      auto [ah, al]   = mul2(s5h, s5l, p, zero(as(p)));
      auto [bh, bl]   = mul2(s3h, s3l, c23h, c23l);
      auto [ch, cl]   = add2(ah, al, bh, bl);
      auto [dh2, dl2] = add2(ch, cl, sh + sh, sl + sl);
      auto [kh, kl]   = mul2(ln2h, ln2l, k, zero(as(k)));
      auto [lh, ll]   = add2(kh, kl, dh2, dl2);
      // y*log|x| = th + tl and exp(th + tl) ~ exp(th) (1 + tl).  Near the ends of the range th
      // is first moved exactly by 64 log(2) towards 0 (tl absorbs the error) and the result
      // scaled back: exp returns infinity in the last binade below valmax, and subnormal results
      // must be rounded after the correction.  frexp is undefined for NaN: restore it.
      constexpr bool d = sizeof(element_type_t<T>) == 8;
      auto [ph, pl] = two_prod(y, lh);
      auto up       = ph > T(d ? 704 : 84);
      auto dn       = ph < T(d ? -704 : -84);
      auto sc       = if_else(up, T(64), if_else(dn, T(-64), zero));
      auto [th, te] = two_add(ph, -sc * ln2h);
      auto tl       = fma(y, ll, pl) + fnma(sc, ln2l, te);
      // Far outside the range of exp, th is clamped: exp converts its scale to an integer, which
      // must stay representable (an emulated conversion out of range is undefined).
      auto z = exp(if_else(is_nan(th), th, eve::clamp(th, T(d ? -1000 : -150), T(d ? 1000 : 150))));
      auto zc       = fma(z, tl, z);
      z             = if_else(is_finite(zc), zc, z);
      z             = z * if_else(up, T(0x1p64), if_else(dn, T(0x1p-64), one));
      return if_else(is_nan(ax), ax, z);
    }

    template<floating_scalar_value T,  integral_scalar_value U, callable_options O>
    EVE_FORCEINLINE constexpr T
    pow_abs_(EVE_REQUIRES(cpu_), O const & o, T a0, U a1) noexcept
    {
      return pow_abs[o](a0, T(a1));
    }

    template<typename T,  typename U, callable_options O>
    EVE_FORCEINLINE constexpr common_value_t<T, U>
    pow_abs_(EVE_REQUIRES(cpu_), O const & o, T a, U b) noexcept
    {
      using r_t = common_value_t<T, U>;
      if constexpr(std::same_as<element_type_t<r_t>, eve::float16_t>)
      {
        return  eve::_::apply_fp16_as_fp32(eve::pow_abs[o], a, b);
      }
      else
      {
        auto x =  r_t(a);
        auto y =  r_t(b);
        if constexpr( scalar_value<r_t> )
          return std::pow(eve::abs(x), y);
        else
        {
          using i_t              = as_integer_t<r_t, unsigned>;
          using eli_t            = element_type_t<i_t>;
          auto        iseqzx     = is_eqz(x);
          auto        ylt0       = y < zero(as(y));
          auto        ax         = eve::abs(x);
          auto        ax_is1     = ax == eve::one(as(x));
          eli_t const largelimit = (sizeof(eli_t) == 4 ? 31 : 63);
          auto [yf, yi]          = eve::modf(eve::abs(y));
          auto test              = yf > r_t(0.5);
          yf                     = dec[test](yf);
          auto lx                = eve::log[o](ax);
          auto z                 = eve::exp[o](yf*lx);
          yi                     = inc[test](yi);
          yi                     = if_else(ax_is1, eve::one, yi);
          auto large             = yi > r_t(largelimit);
          yi                     = if_else(large, eve::one, yi);

          auto russian = [](auto base, auto expo){
            r_t result(1);
            while( eve::any(is_nez(expo)) )
            {
              result *= if_else(is_odd(expo), base, T(1));
              expo = (expo >> 1);
              base = sqr(base);
            }
            return result;
          };
          if constexpr (!O::contains(raw)) yi = if_else(is_not_finite(yi), eve::zero, yi);
          z *= russian(ax, convert(yi, uint_from<T>()));
          z = if_else(iseqzx && ylt0, zero, z);
          z = if_else(is_infinite(ax), inf(as(x)), z);
          z = if_else(ylt0, rec[pedantic](z), z);
          z = if_else(ax_is1 || is_eqz(y), one, z);
          z = if_else(iseqzx && is_gtz(y), zero, z);
          z = if_else(is_nan(x) && is_nan(y), allbits, z);
          // Lanes with a normal x where |x|^yi or |x|^|y| (formed before the reciprocal) may
          // leave the normal range although the result need not also take the helper.  This is
          // decided here, so that the binary powering does not wait for the logarithm.  The raw
          // and fast logarithms are within 3% for |log|x|| in [1, 80], but saturate near 88 for
          // double: route every double lane beyond 80 then.  The range test on x is only needed
          // when some lane is selected.
          auto alx = eve::abs(lx);
          if constexpr( (O::contains(raw) || O::contains(fast)) && sizeof(eli_t) == 8 )
            alx = if_else(alx > r_t(80), inf(as(alx)), alx);
          // yi + 1/2 bounds both yi and |y|, and yi = 1 for unit bases keeps them out.
          auto wider = (yi + r_t(0.5)) * alx > r_t(sizeof(eli_t) == 4 ? 78 : 637);
          // The binary powering above only handles integral parts up to largelimit, and only
          // without leaving the normal range.  The other lanes may still have a finite non-zero
          // result (e.g. (1+2^-30)^64), so they are recomputed from exp and an extended
          // precision log, which also gives their 0, infinity and NaN special cases (where the
          // cases above apply to such a lane, they give the same value).
          if( eve::any(large || wider) )
          {
            large = large || (wider && (ax >= smallestposval(as(ax))) && (ax <= valmax(as(ax))));
            z     = if_else(large, pow_abs_large(ax, y), z);
          }
          return z;
        }
      }
    }
  }
}
