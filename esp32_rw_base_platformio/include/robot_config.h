#pragma once

#include <Arduino.h>

namespace robot_config {

enum class ControlMode : uint8_t {
    kHilBridge = 0,
    kOnboardWheelOnly = 1,
};

constexpr ControlMode kControlMode = ControlMode::kHilBridge;

constexpr bool kUseWifi = true;
constexpr bool kSendTelemetry = true;
constexpr bool kBaseWheelEnabledInHil = true;
constexpr bool kBaseWheelEnabledInOnboard = false;
constexpr bool kHasBaseWheelEncoder = false;

constexpr char kWifiSsid[] = "YOUR_WIFI_SSID";
constexpr char kWifiPassword[] = "YOUR_WIFI_PASSWORD";
constexpr char kPcIp[] = "192.168.1.100";
constexpr uint16_t kPcTelemetryPort = 5005;
constexpr uint16_t kEspCommandPort = 5006;
constexpr uint32_t kWifiConnectTimeoutMs = 10000u;

constexpr uint32_t kLoopHz = 250u;
constexpr uint32_t kLoopPeriodUs = 1000000u / kLoopHz;
constexpr uint32_t kCommandTimeoutUs = 150000u;
constexpr uint32_t kDebugPeriodMs = 1000u;

constexpr float kBatteryNominalV = 11.1f;
constexpr float kBatteryFullV = 12.6f;
constexpr bool kBatterySenseEnabled = false;
constexpr float kBatteryUndervoltageV = 10.2f;
constexpr float kBatteryDividerScale = 1.0f;
constexpr float kBatteryAdcRefV = 3.3f;
constexpr uint16_t kBatteryAdcMax = 4095u;

constexpr int kReactionWheelPolePairs = 7;  // TODO: confirm the real 2804 pole-pair count.
constexpr float kReactionSupplyVoltage = 11.1f;
constexpr float kReactionVoltageLimitHil = 3.0f;
constexpr float kReactionVoltageLimitOnboard = 2.0f;
constexpr float kReactionHilNormToVoltage = 3.0f;
constexpr float kBasePwmLimitHil = 0.45f;
constexpr float kBasePwmLimitOnboard = 0.20f;
constexpr uint32_t kBasePwmFrequencyHz = 20000u;
constexpr uint8_t kBasePwmResolutionBits = 8u;

constexpr float kComplementaryAlpha = 0.98f;
constexpr float kAccelPitchSign = 1.0f;
constexpr float kAccelRollSign = 1.0f;
constexpr float kGyroPitchSign = 1.0f;
constexpr float kGyroRollSign = 1.0f;
constexpr float kReactionSpeedSign = 1.0f;

constexpr bool kInvertReactionCommand = false;
constexpr bool kInvertBaseCommand = false;

constexpr float kPitchEstopDeg = 35.0f;
constexpr float kRollEstopDeg = 35.0f;

constexpr float DegreesToRadians(float deg) {
    return deg * 0.01745329251994329577f;
}

constexpr const char* ControlModeName(ControlMode mode) {
    switch (mode) {
        case ControlMode::kHilBridge:
            return "hil_bridge";
        case ControlMode::kOnboardWheelOnly:
            return "onboard_wheel_only";
        default:
            return "unknown";
    }
}

constexpr bool IsHilMode() {
    return kControlMode == ControlMode::kHilBridge;
}

}  // namespace robot_config

