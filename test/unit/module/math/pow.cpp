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

//==================================================================================================
// Types tests
//==================================================================================================
TTS_CASE_TPL("Check return types of pow", eve::test::simd::ieee_reals)
<typename T>(tts::type<T>)
{
  using v_t = eve::element_type_t<T>;

  TTS_EXPR_IS(eve::pow(T(), T()), T);
  TTS_EXPR_IS(eve::pow(v_t(), v_t()), v_t);
  TTS_EXPR_IS(eve::pow(T(), v_t()), T);
  TTS_EXPR_IS(eve::pow(T(), int()), T);
  TTS_EXPR_IS(eve::pow(v_t(), int()), v_t);
};

//==================================================================================================
// pow  tests
//==================================================================================================
TTS_CASE_WITH("Check behavior of pow on wide",
              eve::test::simd::ieee_reals,
              tts::generate(tts::randoms(1.0, 10.0), tts::randoms(-1.0, 1.0)))
<typename T>(T const& a0, T const& a1)
{
  using v_t = eve::element_type_t<T>;

  TTS_RELATIVE_EQUAL(eve::pow(a0, a1),
                     tts::map([](auto e, auto f) -> v_t { return std::pow(std::abs(e), f); }, a0, a1),
                     0.001);
};

TTS_CASE_TPL("Check  pow", eve::test::simd::all_types)
<typename T>(tts::type<T>)
{
  if constexpr( eve::floating_value<T> )
  {
    TTS_EXPECT(eve::all(eve::is_positive(eve::pow(eve::minf(eve::as<T>()), T(-4)))));
    TTS_EXPECT(eve::all(eve::is_negative(eve::pow(eve::minf(eve::as<T>()), T(-3)))));

    TTS_IEEE_EQUAL(eve::pow(T(0), T(-1)), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(-T(0), T(-1)), eve::minf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(-T(0), T(-2)), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(T(0), T(-2)), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(T(0), eve::minf(eve::as<T>())), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(-T(0), eve::minf(eve::as<T>())), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(-T(1), eve::minf(eve::as<T>())), T(1));
    TTS_IEEE_EQUAL(eve::pow(-T(1), eve::inf(eve::as<T>())), T(1));
    TTS_IEEE_EQUAL(eve::pow(T(1), eve::nan(eve::as<T>())), T(1));
    TTS_IEEE_EQUAL(eve::pow(eve::nan(eve::as<T>()), T(0)), T(1));
    TTS_IEEE_EQUAL(eve::pow(eve::nan(eve::as<T>()), -T(0)), T(1));
    TTS_IEEE_EQUAL(eve::pow(T(0.5), eve::inf(eve::as<T>())), T(0));
    TTS_IEEE_EQUAL(eve::pow(T(2), eve::inf(eve::as<T>())), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(T(0.5), eve::minf(eve::as<T>())), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(T(2), eve::minf(eve::as<T>())), T(0));
    TTS_IEEE_EQUAL(eve::pow(-T(0.5), eve::inf(eve::as<T>())), T(0));
    TTS_IEEE_EQUAL(eve::pow(-T(2), eve::inf(eve::as<T>())), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(-T(0.5), eve::minf(eve::as<T>())), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(-T(2), eve::minf(eve::as<T>())), T(0));
    TTS_IEEE_EQUAL(eve::pow(eve::minf(eve::as<T>()), -T(3)), T(-0.));
    TTS_IEEE_EQUAL(eve::pow(eve::minf(eve::as<T>()), -T(4)), T(0));
    TTS_IEEE_EQUAL(eve::pow(eve::inf(eve::as<T>()), T(4)), eve::inf(eve::as<T>()));
    TTS_IEEE_EQUAL(eve::pow(eve::inf(eve::as<T>()), -T(4)), T(0));
  }

 TTS_EQUAL(eve::pow(T(2), -3), T(1 / 8.0));
  TTS_EQUAL(eve::pow(T(2), 3), T(8));
  TTS_EQUAL(eve::pow(T(2), 3u), T(8));
  TTS_EQUAL(eve::pow(2, 3), (8));

  using v_t = eve::element_type_t<T>;
  if constexpr( std::is_floating_point_v<v_t> )
  {
    using w8_t = eve::wide<v_t, eve::fixed<8>>;
    auto Inf   = eve::inf(eve::as<v_t>());
    {
      w8_t a(-0.0, -0.0, -0.0, -0.0, -1.0, -1.0, -Inf, -Inf);
      w8_t b(-Inf, -3.0, -4.0, -4.5, -Inf, Inf, -3.0, -4.0);
      w8_t rr(Inf, -Inf, Inf, Inf, 1.0, 1.0, -0.0, 0.0);
      w8_t c = (eve::pow)(a, b);
      TTS_ULP_EQUAL(c, rr, 2);
    }
    {
      using w4_t = eve::wide<v_t, eve::fixed<4>>;
      w4_t a(-Inf, -Inf, -Inf, 0.0);
      w4_t b(-4.5, 4.0, 4.5, 0.0);
      w4_t rr(0.0, Inf, Inf, 1.0);
      w4_t c = (eve::pow)(a, b);
      TTS_ULP_EQUAL(c, rr, 2);
    }
  }
};

//==================================================================================================
// pow with exponents whose integral part is beyond the binary powering range of pow_abs
//==================================================================================================
TTS_CASE_TPL("Check pow with large exponents", eve::test::simd::ieee_reals)
<typename T>(tts::type<T>)
{
  using v_t        = eve::element_type_t<T>;
  constexpr bool d = sizeof(v_t) == 8;
  auto check = [](v_t a, v_t b)
  {
    TTS_ULP_EQUAL(eve::pow(T(a), T(b)), T(std::pow(a, b)), 2) << "x: " << a << " y: " << b << '\n';
  };
  check(v_t(1.0001), v_t(64));
  check(v_t(1.0001), v_t(1000.5));
  check(v_t(-1.0001), v_t(65));
  check(v_t(-1.0001), v_t(64));
  check(v_t(-0.9999), v_t(-1001));
  check(v_t(-2), v_t(d ? 1023 : 127));
  check(v_t(-2), v_t(d ? -1021 : -125));
  check(v_t(0.5), v_t(d ? 1000.5 : 100.5));
  TTS_IEEE_EQUAL(eve::pow(T(-2), T(d ? 1025 : 129)), eve::minf(eve::as<T>()));
  TTS_IEEE_EQUAL(eve::pow(T(-2), T(d ? 1024 : 128)), eve::inf(eve::as<T>()));
  TTS_IEEE_EQUAL(eve::pow(T(-2), T(d ? -1101 : -201)), T(-0.));
  TTS_EXPECT(eve::all(eve::is_negative(eve::pow(T(-2), T(d ? -1101 : -201)))));
  TTS_EXPECT(eve::all(eve::is_positive(eve::pow(T(-2), T(d ? -1100 : -200)))));
  TTS_IEEE_EQUAL(eve::pow(T(-1.0001), T(d ? 1e7 + 1 : 2e6 + 1)), eve::minf(eve::as<T>()));
  TTS_IEEE_EQUAL(eve::pow(T(-1.0001), T(d ? -1e7 : -2e6)), T(0));

  // Lanes needing and not needing the large exponent path in the same vector
  auto x = T([](auto i, auto) { return (i % 2 == 0) ? v_t(-1.0001) : v_t(0.9999); });
  auto y = T([](auto i, auto)
             { return (i % 3 == 0) ? v_t(3) : (i % 3 == 1) ? v_t(65) : v_t(-127); });
  auto expected = tts::map([](auto e, auto f) -> v_t { return std::pow(e, f); }, x, y);
  TTS_ULP_EQUAL(eve::pow(x, y), expected, 8);
};

//==================================================================================================
// Tests for masked pow
//==================================================================================================
TTS_CASE_WITH("Check behavior of eve::masked(eve::pow)(eve::wide)",
              eve::test::simd::ieee_reals,
              tts::generate(tts::randoms(eve::valmin, eve::valmax),
                            tts::randoms(eve::valmin, eve::valmax),
                            tts::logicals(0, 3)))
<typename T, typename M>(T const& a0,
                         T const& a1,
                         M const& mask)
{
  TTS_IEEE_EQUAL(eve::pow[mask](a0, a1),eve::if_else(mask, eve::pow(a0, a1), a0));
};
