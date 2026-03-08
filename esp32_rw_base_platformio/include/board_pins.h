#pragma once

namespace board_pins {

constexpr int kI2cSda = 21;
constexpr int kI2cScl = 22;

constexpr int kDrvEnable = 25;
constexpr int kDrvPhaseU = 26;
constexpr int kDrvPhaseV = 27;
constexpr int kDrvPhaseW = 14;

constexpr int kBtsRpwm = 32;
constexpr int kBtsLpwm = 33;
constexpr int kBtsRen = 18;
constexpr int kBtsLen = 19;

constexpr int kBatterySenseAdc = 34;
constexpr int kStatusLed = 2;

}  // namespace board_pins

