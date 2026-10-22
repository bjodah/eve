//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Project Contributors
  SPDX-License-Identifier: BSL-1.0
**/
//==================================================================================================
#include "test.hpp"

#include <eve/module/core.hpp>
#include <eve/module/math.hpp>

#include <cmath>
#include <concepts>
#include <initializer_list>
#include <utility>

//==================================================================================================
// Types tests
//==================================================================================================
TTS_CASE_TPL("Check return types of pow_abs", eve::test::simd::ieee_reals_wf16)
<typename T>(tts::type<T>)
{
  using v_t = eve::element_type_t<T>;

  TTS_EXPR_IS(eve::pow_abs(T(), T()), T);
  TTS_EXPR_IS(eve::pow_abs(v_t(), v_t()), v_t);
  TTS_EXPR_IS(eve::pow_abs(T(), v_t()), T);
  TTS_EXPR_IS(eve::pow_abs(v_t(), T()), T);
};

template < typename T> T std_pow_abs(T a,  T b)
{
  if constexpr(sizeof(eve::element_type_t<T>) == 2)
  {
    auto cvf = [](auto xx){return eve::convert(xx, eve::as<float>());};
    auto pa = [cvf](auto e, auto f) { return eve::convert(eve::pow_abs(cvf(e), cvf(f)), eve::as<eve::float16_t>()); };
    return tts::map(pa, a, b);
  }
  else
  {
    auto pb = [](auto e, auto f) { return std::pow(std::abs(e), f); };
    return tts::map(pb, a, b);
  }
};

//==================================================================================================
// pow_abs  tests
//==================================================================================================
TTS_CASE_WITH("Check behavior of pow_abs on wide",
              eve::test::simd::ieee_reals_wf16,
              tts::generate(tts::randoms(1, 10),
                            tts::randoms(-1.0, 1.0),
                            tts::randoms(-1.0, 1.0),
                            tts::randoms(-1.0, 1.0)))
<typename T>(T const& a0, T const& a1, T const& a2, T const& a3)
{
  TTS_ULP_EQUAL(eve::pow_abs(a0, a1), std_pow_abs(a0, a1), 32);
  TTS_ULP_EQUAL(eve::pow_abs(a2, a3), std_pow_abs(a2, a3), 32);
};

TTS_CASE_TPL("Check limits of pow_abs", eve::test::simd::ieee_reals_wf16)
<typename T>(tts::type<T>)
{
  using v_t = eve::element_type_t<T>;
  if constexpr( eve::platform::supports_invalids )
  {
    TTS_IEEE_EQUAL(eve::pow_abs(eve::minf(eve::as<T>()), eve::minf(eve::as<T>())), T(0));

    TTS_IEEE_EQUAL(eve::pow_abs(eve::inf(eve::as<T>()), T(-0.5)), T(0));
    TTS_IEEE_EQUAL(eve::pow_abs(eve::inf(eve::as<T>()), T(0.5)), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow_abs(eve::inf(eve::as<T>()), eve::inf(eve::as<T>())),
                   eve::inf(eve::as<T>()));

    TTS_IEEE_EQUAL(eve::pow_abs(eve::nan(eve::as<T>()), eve::nan(eve::as<T>())),
                   eve::nan(eve::as<T>()));

    TTS_IEEE_EQUAL(eve::pow_abs(T(2), eve::minf(eve::as<T>())), T(0));
    TTS_IEEE_EQUAL(eve::pow_abs(T(2), eve::inf(eve::as<T>())), eve::inf(eve::as<T>()));

    TTS_IEEE_EQUAL(eve::pow_abs(T(0), T(-1)), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow_abs(T(0), T(-2)), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow_abs(T(0.5), eve::inf(eve::as<T>())), T(0));
    TTS_IEEE_EQUAL(eve::pow_abs(T(0.5), eve::minf(eve::as<T>())), eve::inf(eve::as<T>()));
  }
  auto hf = eve::half(eve::as<T>());
  auto cv = []<typename U>(U x) {
    if constexpr(sizeof(v_t) == 2) return T(x);
    else return U(x);
  };
  TTS_IEEE_EQUAL(eve::pow_abs(T(-1), T(-1)), T(1));
  TTS_IEEE_EQUAL(eve::pow_abs(T(-1), T(5)), T(1));
  TTS_IEEE_EQUAL(eve::pow_abs(T(-1), T(6)), T(1));
  TTS_ULP_EQUAL(eve::pow_abs(T(0.5), T(0.25)), T(0.840896415253715), 2);
  TTS_ULP_EQUAL(eve::pow_abs(T(0.5), T(0.25)), cv(std_pow_abs(hf, hf/2)), 2);
  TTS_ULP_EQUAL(eve::pow_abs(T(0.5), T(2.5)), cv(std_pow_abs(hf, 2+hf)), 2);
  TTS_ULP_EQUAL(eve::pow_abs(T(1.5), T(1.0e19)), eve::inf(eve::as<T>()), 2);
  TTS_ULP_EQUAL(eve::pow_abs(T(0.5), T(1.0e19)), T(0), 2);
  TTS_ULP_EQUAL(eve::pow_abs(T(10.0), T(10.0)), T(10000000000.0), 2);

  if constexpr(sizeof(v_t) > 2)
  {
    using w8_t = eve::wide<v_t, eve::fixed<8>>;
    {
      w8_t a(0.25, 0.5, 1, 2.5, 100.0, 12.7, 4.0, 0.5);
      w8_t b(1.1, 2.2, 3.3, 4.4, 43.2, -0.7, 0.4, 0.0);
      TTS_ULP_EQUAL(eve::pow_abs(a, b), eve::pow_abs(a, b), 2);
    }
    auto Nan = eve::nan(eve::as<v_t>());
    auto Inf = eve::inf(eve::as<v_t>());
    {
      w8_t a(2.4, 2.4, 1.0, 23.5, Nan, 12.5, 0.0, 0.0);
      w8_t b(0.0, -0.0, 2.4, 1.0, 12.5, Nan, -3.0, -3.5);
      w8_t rr(1.0, 1.0, 1.0, 23.5, Nan, Nan, Inf, Inf);
      w8_t c = eve::pow_abs(a, b);
      TTS_ULP_EQUAL(c, rr, 2);
    }
    {
      w8_t a(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0);
      w8_t b(-4.0, -Inf, -Inf, 3.5, 3.0, 4.0, Inf, -Inf);
      w8_t rr(Inf, Inf, Inf, 0.0, 0.0, 0.0, 1.0, 1.0);
      w8_t c = eve::pow_abs(a, b);
      TTS_ULP_EQUAL(c, rr, 2);
    }
    {
      w8_t a(2.0, 2.0, Nan, Inf, Inf, 1.0, 0.5, 1.5);
      w8_t b(0.0, -0.0, 0.0, 0.0, 0.0, -Inf, Inf, Inf);
      w8_t rr(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, Inf);
      w8_t c = eve::pow_abs(a, b);
      TTS_ULP_EQUAL(c, rr, 2);
    }
    {
      using w4_t = eve::wide<v_t, eve::fixed<4>>;
      w4_t a(0.5, 1.5, Inf, Inf);
      w4_t b(-Inf, -Inf, -3.0, 3.0);
      w4_t rr(Inf, 0.0, 0.0, Inf);
      w4_t c = eve::pow_abs(a, b);
      TTS_ULP_EQUAL(c, rr, 2);
    }
  }
};


//==================================================================================================
// pow_abs with exponents whose integral part is beyond the binary powering range (31 for float,
// 63 for double): the result is finite whenever |y log|x|| is small enough.
//==================================================================================================
TTS_CASE_TPL("Check pow_abs with large exponents", eve::test::simd::ieee_reals)
<typename T>(tts::type<T>)
{
  using v_t        = eve::element_type_t<T>;
  constexpr bool d = sizeof(v_t) == 8;
  auto ref         = [](v_t a, v_t b) { return std::pow(std::abs(a), b); };
  auto limit       = v_t(d ? 63 : 31);

  v_t const n = d ? v_t(0x1p-30) : v_t(0x1p-20);
  v_t const bases[] = {1 + n, 1 - n, v_t(1.0001), v_t(0.9999), v_t(-1.0001), v_t(1.5), v_t(0.5),
                       v_t(2), v_t(3), v_t(7.25)};
  v_t const exps[]  = {31, 32, 33, 63, 64, 65, 127, 128, v_t(31.6), v_t(63.6), v_t(64.5),
                       v_t(1000.5), v_t(12345.75), v_t(1e5), -31, -32, -63, -64, -65, -127,
                       -128, v_t(-64.5), v_t(-1000.5), v_t(-1e5)};
  for( auto a : bases )
  {
    for( auto b : exps )
    {
      auto   expected = ref(a, b);
      double tol      = std::abs(b) > limit ? 2 : 8;
      if( expected == 0 || std::isinf(expected) )
      {
        TTS_IEEE_EQUAL(eve::pow_abs(T(a), T(b)), T(expected)) << "x: " << a << " y: " << b << '\n';
      }
      else
      {
        TTS_ULP_EQUAL(eve::pow_abs(T(a), T(b)), T(expected), tol)
          << "x: " << a << " y: " << b << '\n';
      }
    }
  }

  // Each helper below instantiates the kernel once per decorator: the cases are tables.
  auto checkd = [&](auto p, v_t a, v_t b, double ulps)
  {
    TTS_ULP_EQUAL(p(T(a), T(b)), T(ref(a, b)), ulps) << "x: " << a << " y: " << b << '\n';
  };
  auto check = [&](v_t a, v_t b, double ulps) { checkd(eve::pow_abs, a, b, ulps); };
  struct exact_case { v_t x, y, r; };
  auto exact = [&](auto p, std::initializer_list<exact_case> cases)
  {
    for( auto c : cases )
      TTS_IEEE_EQUAL(p(T(c.x), T(c.y)), T(c.r)) << "x: " << c.x << " y: " << c.y << '\n';
  };
  auto const inf  = eve::inf(eve::as<v_t>());
  auto const nan  = eve::nan(eve::as<v_t>());
  auto const vmax = eve::valmax(eve::as<v_t>());
  auto const vmin = eve::valmin(eve::as<v_t>());
  auto const spv  = eve::smallestposval(eve::as<v_t>());
  auto const eps  = eve::eps(eve::as<v_t>());
  auto const dmin = eve::mindenormal(eve::as<v_t>());
  auto const e0   = v_t(d ? -1074 : -149);

  // Bases close to one with exponents up to the range limits, huge finite exponents, and the
  // last binade below valmax
  for( auto c : std::initializer_list<std::pair<v_t, v_t>>{
         {v_t(1.0001), v_t(d ? 7e6 : 8e5)}, {v_t(0.9999), v_t(d ? 7e6 : 8e5)},
         {v_t(1.0001), v_t(d ? -7e6 : -8e5)}, {1 + n, v_t(d ? 0x1p38 : 0x1p25)},
         {1 - n, v_t(d ? -0x1p38 : -0x1p25)}, {v_t(2), v_t(d ? 1023.5 : 127.5)},
         {1 + eps, v_t(d ? 0x1p60 : 0x1p28)}, {1 - eps / 2, v_t(d ? -0x1p61 : -0x1p29)}} )
    check(c.first, c.second, 2);
  check(v_t(2), v_t(d ? -1022 : -126), 1);
  check(v_t(4), v_t(d ? 64.5 : 32.5), 1);

  // Genuine overflow and underflow, and zero, infinite and NaN bases
  exact(eve::pow_abs, {{v_t(2), v_t(d ? 1024 : 128), inf},
                       {1 + eps, v_t(d ? 0x1p62 : 0x1p30), inf},
                       {1 + eps, vmax, inf},
                       {v_t(0.9999), vmax, 0},
                       {v_t(2), vmin, 0},
                       {v_t(1.0001), v_t(d ? 1e7 : 2e6), inf},
                       {v_t(1.0001), v_t(d ? -1e7 : -2e6), 0},
                       {v_t(0.9999), v_t(d ? 1e7 : 2e6), 0},
                       {v_t(0.9999), v_t(d ? -1e7 : -2e6), inf},
                       {v_t(10), v_t(d ? 400 : 40), inf},
                       {v_t(10), v_t(d ? -400 : -50), 0},
                       {vmax, v_t(64.5), inf},
                       {spv, v_t(64.5), 0},
                       {spv / 4, v_t(-64.5), inf},
                       {v_t(0), v_t(-64.5), inf},
                       {v_t(-0.), v_t(-127), inf},
                       {inf, v_t(-64.5), 0},
                       {-inf, v_t(64.5), inf},
                       {v_t(2), v_t(d ? 1e300 : 1e38), inf},
                       {v_t(2), v_t(d ? -1e300 : -1e38), 0},
                       {v_t(0.5), v_t(d ? 1e300 : 1e38), 0},
                       {v_t(d ? 0x1p700 : 0x1p96), v_t(-1.5), v_t(d ? 0x1p-1050 : 0x1p-144)}});

  // Gradual underflow, including the tie 2^(emin-1) between 0 and the smallest denormal, which
  // rounds to zero; NaN bases; unit bases with infinite exponents
  auto specials = [&](auto p)
  {
    exact(p, {{v_t(2), e0 + 1, 2 * dmin}, {v_t(2), e0, dmin}, {v_t(2), e0 - v_t(0.5), dmin},
              {v_t(2), e0 - 1, 0}, {v_t(0.5), 1 - e0, 0}, {v_t(2), e0 - 2, 0}});
    for( auto b : {v_t(64.5), v_t(-64.5), v_t(128), v_t(-1e5), inf, -inf} )
      TTS_IEEE_EQUAL(p(T(nan), T(b)), T(nan)) << "y: " << b << '\n';
    // NaN exponents, including bases whose |log x| sends fast double lanes to the large
    // exponent path (raw and fast exp do not keep NaN on the regular path, and raw keeps the
    // NaN integral part); huge finite exponents must not overflow exp's integer scale
    constexpr bool is_raw  = std::same_as<decltype(p), decltype(eve::pow_abs[eve::raw])>;
    constexpr bool is_fast = std::same_as<decltype(p), decltype(eve::pow_abs[eve::fast])>;
    if constexpr( !is_raw && (!is_fast || d) )
    {
      for( auto a : {spv, vmax, v_t(2), v_t(0.5)} )
      {
        if( is_fast && (a == v_t(2) || a == v_t(0.5)) ) continue;
        TTS_IEEE_EQUAL(p(T(a), T(nan)), T(nan)) << "x: " << a << '\n';
        TTS_IEEE_EQUAL(p(T(-a), T(nan)), T(nan)) << "x: " << -a << '\n';
      }
    }
    TTS_IEEE_EQUAL(p(T(2), T(vmax)), T(inf));
    TTS_IEEE_EQUAL(p(T(2), T(-vmax)), T(0));
    for( auto a : {v_t(1), v_t(-1)} )
    {
      for( auto b : {inf, -inf, nan, v_t(64.5), v_t(-1e5)} )
        TTS_IEEE_EQUAL(p(T(a), T(b)), T(1)) << "x: " << a << " y: " << b << '\n';
    }
    // The same unit bases next to lanes taking the large exponent path
    auto xm = T([](auto i, auto)
                { return (i % 3 == 0) ? v_t(1) : (i % 3 == 1) ? v_t(-1) : v_t(1.5); });
    auto ym = T([](auto i, auto) { return (i % 2 == 0) ? eve::inf(eve::as<v_t>()) : v_t(-100.5); });
    TTS_ULP_EQUAL(p(xm, ym), tts::map([](auto e, auto f) -> v_t
                                      { return std::abs(e) == 1 ? v_t(1) : std::pow(e, f); },
                                      xm, ym), 2);
  };
  specials(eve::pow_abs);
  specials(eve::pow_abs[eve::pedantic]);
  specials(eve::pow_abs[eve::raw]);
  specials(eve::pow_abs[eve::fast]);
  {
    // Even lanes alternate 2^emin and the tie, odd lanes stay on the binary powering path
    auto xu = T(v_t(2));
    auto yu = T([e0](auto i, auto) { return (i % 2 == 0) ? e0 - v_t((i % 4) / 2) : v_t(3); });
    TTS_IEEE_EQUAL(eve::pow_abs(xu, yu), tts::map(ref, xu, yu));
  }

  // Decorators share the kernel, and the large exponent lanes take the same accurate path
  checkd(eve::pow_abs[eve::pedantic], v_t(1.0001), v_t(64), 2);
  checkd(eve::pow_abs[eve::raw], v_t(1.0001), v_t(64.5), 2);
  checkd(eve::pow_abs[eve::fast], v_t(1.0001), v_t(64.5), 2);
  checkd(eve::pow_abs[eve::fast], v_t(0.9999), v_t(-1e5), 2);

  // Bases at the edges of the log reduction (mantissas just below 1.5 and 0.75), with
  // |y log|x|| close to the overflow and underflow thresholds
  for( v_t a : {std::nextafter(v_t(1.5), v_t(0)), std::nextafter(v_t(0.75), v_t(0)), v_t(1.4),
                v_t(0.74), std::nextafter(v_t(3), v_t(0)), std::nextafter(v_t(0.375), v_t(0))} )
  {
    for( v_t t : {v_t(d ? 709.0 : 88.0), v_t(d ? 709.7 : 88.6), v_t(d ? -708.0 : -87.0)} )
      check(a, t / std::log(a), 2);
  }

  // Small exponents whose binary power |x|^yi, or |x|^|y| before the reciprocal, leaves the
  // normal range although the result does not
  for( auto p : {v_t(1), v_t(-1)} )
  {
    check(v_t(d ? std::exp(150.) : std::exp(10.f)), p * v_t(d ? 4.51 : 8.51), 2);
    check(v_t(d ? std::exp(-150.) : std::exp(-10.f)), p * v_t(d ? 4.51 : 8.51), 2);
    check(v_t(d ? 94186.28 : 22026.47f), p * v_t(d ? 61.51 : 8.4), 2);
  }
  check(v_t(d ? std::exp(600.) : std::exp(70.f)), v_t(d ? -1.2 : -1.3), 2);
  check(v_t(d ? std::exp(300.) : std::exp(35.f)), v_t(d ? -2.45 : -2.6), 2);
  if constexpr( !d )
  {
    check(v_t(23.78), v_t(-27.54), 2);
    check(v_t(0x1.fbfd22p+100f), v_t(-1.3), 2);
  }
  {
    auto xs = T([](auto i, auto) { return (i % 2 == 0) ? v_t(d ? 0x1p700 : 0x1p96) : v_t(3); });
    auto ys = T([](auto i, auto) { return (i % 2 == 0) ? v_t(-1.5) : v_t(2); });
    auto zs = eve::pow_abs(xs, ys);
    TTS_EXPECT(eve::all(eve::is_nez(zs) && eve::is_finite(zs)));
    TTS_IEEE_EQUAL(zs, tts::map(ref, xs, ys));
  }

  // The same with the raw and fast logarithms, which are not accurate over the whole range
  for( auto b : {v_t(1.7), v_t(-1.7)} )
  {
    v_t a = d ? v_t(std::exp(400.)) : v_t(std::exp(50.f));
    checkd(eve::pow_abs[eve::raw], a, b, 2);
    checkd(eve::pow_abs[eve::fast], a, b, 2);
  }

  // Lanes needing and not needing the large exponent path in the same vector
  auto x = T([](auto i, auto)
             { return (i % 3 == 0) ? v_t(0.9999) : (i % 3 == 1) ? v_t(1.5) : v_t(1.0001); });
  auto y = T([](auto i, auto) { return (i % 2 == 0) ? v_t(2.5) : v_t(-1000.5); });
  TTS_ULP_EQUAL(eve::pow_abs(x, y), tts::map(ref, x, y), 2);
};

//==================================================================================================
// Tests for masked pow_abs
//==================================================================================================
TTS_CASE_WITH("Check behavior of eve::maskedeve::pow_abs(eve::wide)",
              eve::test::simd::ieee_reals_wf16,
              tts::generate(tts::randoms(eve::valmin, eve::valmax),
                            tts::randoms(eve::valmin, eve::valmax),
                            tts::logicals(0, 3)))
<typename T, typename M>(T const& a0,
                         T const& a1,
                         M const& mask)
{
  TTS_IEEE_EQUAL(eve::pow_abs[mask](a0, a1),
            eve::if_else(mask, eve::pow_abs(a0, a1), a0));
};
