#include <Arduino.h>
#include <ArduinoJson.h>
#include <SimpleFOC.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

#include <math.h>
#include <string.h>

#include "board_pins.h"
#include "robot_config.h"

extern "C" {
#include "actuator_guard.h"
#include "control_loop.h"
#include "controller.h"
#include "controller_params.h"
#include "estimator.h"
#include "fault_manager.h"
}

namespace {

constexpr uint8_t kBmi088AccelAddr = 0x19;
constexpr uint8_t kBmi088GyroAddr = 0x69;
constexpr float kAccelScaleMps2 = 9.81f * 6.0f / 32768.0f;
constexpr float kGyroScaleDps = 2000.0f / 32768.0f;
constexpr uint8_t kBasePwmChannelForward = 0;
constexpr uint8_t kBasePwmChannelReverse = 1;

struct RawImuSample {
    float ax_m_s2 = 0.0f;
    float ay_m_s2 = 0.0f;
    float az_m_s2 = 0.0f;
    float gx_rad_s = 0.0f;
    float gy_rad_s = 0.0f;
    float gz_rad_s = 0.0f;
    bool valid = false;
};

struct AttitudeEstimate {
    float pitch_rad = 0.0f;
    float roll_rad = 0.0f;
    float pitch_rate_rad_s = 0.0f;
    float roll_rate_rad_s = 0.0f;
    bool valid = false;
};

struct ReactionWheelSample {
    float angle_rad = 0.0f;
    float speed_rad_s = 0.0f;
    bool valid = false;
};

struct BaseWheelEncoderSample {
    float pos_m = 0.0f;
    float vel_m_s = 0.0f;
    bool valid = false;
};

struct BridgeCommand {
    float reaction_norm = 0.0f;
    float base_norm = 0.0f;
    bool estop = true;
    bool valid = false;
    uint32_t last_rx_us = 0u;
    uint32_t seq = 0u;
};

struct AppliedOutputs {
    float reaction_norm = 0.0f;
    float base_norm = 0.0f;
    bool estop = true;
};

WiFiUDP g_udp;
IPAddress g_pc_ip;
bool g_wifi_ready = false;
uint32_t g_last_tick_us = 0u;
uint32_t g_last_debug_ms = 0u;
uint32_t g_telem_seq = 0u;

MagneticSensorI2C g_reaction_sensor = MagneticSensorI2C(AS5600_I2C);
BLDCMotor g_reaction_motor = BLDCMotor(robot_config::kReactionWheelPolePairs);
BLDCDriver3PWM g_reaction_driver = BLDCDriver3PWM(
    board_pins::kDrvPhaseU,
    board_pins::kDrvPhaseV,
    board_pins::kDrvPhaseW,
    board_pins::kDrvEnable
);

controller_state_t g_controller_state;
estimator_state_t g_estimator_state;
fault_state_t g_fault_state;
control_loop_safety_state_t g_safety_state;

BridgeCommand g_bridge_command;
AppliedOutputs g_applied_outputs;
float g_u_eff_applied[3] = {0.0f, 0.0f, 0.0f};
float g_reaction_target_voltage = 0.0f;
float g_pitch_state_rad = 0.0f;
float g_roll_state_rad = 0.0f;
volatile int32_t g_base_encoder_counts = 0;
volatile uint8_t g_base_encoder_state = 0u;
int32_t g_last_base_encoder_counts = 0;

float clampf(float value, float lo, float hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

bool wifi_credentials_configured() {
    return strcmp(robot_config::kWifiSsid, "YOUR_WIFI_SSID") != 0 &&
           strcmp(robot_config::kWifiPassword, "YOUR_WIFI_PASSWORD") != 0;
}

void i2c_write(uint8_t addr, uint8_t reg, uint8_t value) {
    Wire.beginTransmission(addr);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

void init_bmi088() {
    i2c_write(kBmi088AccelAddr, 0x7C, 0x04);
    delay(5);
    i2c_write(kBmi088AccelAddr, 0x40, 0xA8);

    i2c_write(kBmi088GyroAddr, 0x0F, 0x00);
    i2c_write(kBmi088GyroAddr, 0x10, 0x07);
    delay(10);
}

RawImuSample read_bmi088() {
    RawImuSample sample;
    uint8_t buf[7] = {0};

    Wire.beginTransmission(kBmi088AccelAddr);
    Wire.write(0x12);
    if (Wire.endTransmission(false) == 0) {
        const int count = Wire.requestFrom(static_cast<int>(kBmi088AccelAddr), 7);
        if (count == 7) {
            Wire.read();
            for (int i = 0; i < 6; ++i) {
                buf[i] = static_cast<uint8_t>(Wire.read());
            }
            const int16_t raw_ax = static_cast<int16_t>((buf[1] << 8) | buf[0]);
            const int16_t raw_ay = static_cast<int16_t>((buf[3] << 8) | buf[2]);
            const int16_t raw_az = static_cast<int16_t>((buf[5] << 8) | buf[4]);
            sample.ax_m_s2 = static_cast<float>(raw_ax) * kAccelScaleMps2;
            sample.ay_m_s2 = static_cast<float>(raw_ay) * kAccelScaleMps2;
            sample.az_m_s2 = static_cast<float>(raw_az) * kAccelScaleMps2;
            sample.valid = true;
        }
    }

    Wire.beginTransmission(kBmi088GyroAddr);
    Wire.write(0x02);
    if (Wire.endTransmission(false) == 0) {
        const int count = Wire.requestFrom(static_cast<int>(kBmi088GyroAddr), 6);
        if (count == 6) {
            for (int i = 0; i < 6; ++i) {
                buf[i] = static_cast<uint8_t>(Wire.read());
            }
            const int16_t raw_gx = static_cast<int16_t>((buf[1] << 8) | buf[0]);
            const int16_t raw_gy = static_cast<int16_t>((buf[3] << 8) | buf[2]);
            const int16_t raw_gz = static_cast<int16_t>((buf[5] << 8) | buf[4]);
            sample.gx_rad_s = radians(static_cast<float>(raw_gx) * kGyroScaleDps);
            sample.gy_rad_s = radians(static_cast<float>(raw_gy) * kGyroScaleDps);
            sample.gz_rad_s = radians(static_cast<float>(raw_gz) * kGyroScaleDps);
        } else {
            sample.valid = false;
        }
    } else {
        sample.valid = false;
    }

    return sample;
}

AttitudeEstimate update_attitude(const RawImuSample& imu, float dt_s) {
    AttitudeEstimate attitude;
    if (!imu.valid) {
        return attitude;
    }

    const float accel_pitch = atan2f(
        robot_config::kAccelPitchSign * imu.ax_m_s2,
        sqrtf(imu.ay_m_s2 * imu.ay_m_s2 + imu.az_m_s2 * imu.az_m_s2)
    );
    const float accel_roll = atan2f(
        robot_config::kAccelRollSign * imu.ay_m_s2,
        sqrtf(imu.ax_m_s2 * imu.ax_m_s2 + imu.az_m_s2 * imu.az_m_s2)
    );
    const float pitch_rate = robot_config::kGyroPitchSign * imu.gy_rad_s;
    const float roll_rate = robot_config::kGyroRollSign * imu.gx_rad_s;

    g_pitch_state_rad = robot_config::kComplementaryAlpha * (g_pitch_state_rad + pitch_rate * dt_s) +
                        (1.0f - robot_config::kComplementaryAlpha) * accel_pitch;
    g_roll_state_rad = robot_config::kComplementaryAlpha * (g_roll_state_rad + roll_rate * dt_s) +
                       (1.0f - robot_config::kComplementaryAlpha) * accel_roll;

    attitude.pitch_rad = g_pitch_state_rad;
    attitude.roll_rad = g_roll_state_rad;
    attitude.pitch_rate_rad_s = pitch_rate;
    attitude.roll_rate_rad_s = roll_rate;
    attitude.valid = true;
    return attitude;
}

ReactionWheelSample read_reaction_wheel() {
    ReactionWheelSample sample;
    g_reaction_sensor.update();
    sample.angle_rad = g_reaction_sensor.getAngle();
    sample.speed_rad_s = robot_config::kReactionSpeedSign * g_reaction_sensor.getVelocity();
    sample.valid = true;
    return sample;
}

void IRAM_ATTR handle_base_encoder_edge() {
    const uint8_t prev = g_base_encoder_state;
    const uint8_t curr =
        (static_cast<uint8_t>(digitalRead(board_pins::kBaseEncoderA)) << 1) |
        static_cast<uint8_t>(digitalRead(board_pins::kBaseEncoderB));
    static const int8_t kTransitionLut[16] = {0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0};
    const int8_t step = kTransitionLut[(prev << 2) | curr];
    if (robot_config::kInvertBaseEncoder) {
        g_base_encoder_counts -= static_cast<int32_t>(step);
    } else {
        g_base_encoder_counts += static_cast<int32_t>(step);
    }
    g_base_encoder_state = curr;
}

void begin_base_encoder() {
    if (!robot_config::kHasBaseWheelEncoder) {
        return;
    }
    const uint8_t pin_mode = robot_config::kBaseEncoderUsePullups ? INPUT_PULLUP : INPUT;
    pinMode(board_pins::kBaseEncoderA, pin_mode);
    pinMode(board_pins::kBaseEncoderB, pin_mode);
    g_base_encoder_state =
        (static_cast<uint8_t>(digitalRead(board_pins::kBaseEncoderA)) << 1) |
        static_cast<uint8_t>(digitalRead(board_pins::kBaseEncoderB));
    attachInterrupt(digitalPinToInterrupt(board_pins::kBaseEncoderA), handle_base_encoder_edge, CHANGE);
    attachInterrupt(digitalPinToInterrupt(board_pins::kBaseEncoderB), handle_base_encoder_edge, CHANGE);
}

BaseWheelEncoderSample read_base_encoder(float dt_s) {
    BaseWheelEncoderSample sample;
    if (!robot_config::kHasBaseWheelEncoder) {
        return sample;
    }

    noInterrupts();
    const int32_t counts = g_base_encoder_counts;
    interrupts();

    const float meters_per_count =
        (2.0f * PI * robot_config::kBaseWheelRadiusM) / fmaxf(static_cast<float>(robot_config::kBaseEncoderTicksPerRev), 1.0f);
    const int32_t delta_counts = counts - g_last_base_encoder_counts;
    g_last_base_encoder_counts = counts;

    sample.pos_m = static_cast<float>(counts) * meters_per_count;
    sample.vel_m_s = static_cast<float>(delta_counts) * meters_per_count / fmaxf(dt_s, 1.0e-4f);
    sample.valid = true;
    return sample;
}

void clear_faults_and_states() {
    controller_reset(&g_controller_state);
    estimator_reset(&g_estimator_state);
    fault_manager_reset(&g_fault_state);
    control_loop_safety_reset(&g_safety_state);
    memset(g_u_eff_applied, 0, sizeof(g_u_eff_applied));
    g_reaction_target_voltage = 0.0f;
    g_applied_outputs = AppliedOutputs{};
    g_bridge_command = BridgeCommand{};
    g_last_base_encoder_counts = 0;
}

void begin_wifi_if_needed() {
    if (!robot_config::kUseWifi || !wifi_credentials_configured()) {
        Serial.println("WiFi disabled or credentials not configured.");
        return;
    }

    WiFi.mode(WIFI_STA);
    WiFi.begin(robot_config::kWifiSsid, robot_config::kWifiPassword);
    Serial.print("WiFi");
    const uint32_t start_ms = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - start_ms > robot_config::kWifiConnectTimeoutMs) {
            Serial.println();
            Serial.println("WiFi connect timeout. Continuing offline.");
            return;
        }
        delay(250);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("WiFi connected: ");
    Serial.println(WiFi.localIP());

    if (!g_pc_ip.fromString(robot_config::kPcIp)) {
        Serial.println("Invalid PC IP string. Telemetry disabled.");
        return;
    }

    g_udp.begin(robot_config::kEspCommandPort);
    g_wifi_ready = true;
}

void begin_base_motor_driver() {
    pinMode(board_pins::kBtsRen, OUTPUT);
    pinMode(board_pins::kBtsLen, OUTPUT);
    digitalWrite(board_pins::kBtsRen, HIGH);
    digitalWrite(board_pins::kBtsLen, HIGH);

    ledcSetup(kBasePwmChannelForward, robot_config::kBasePwmFrequencyHz, robot_config::kBasePwmResolutionBits);
    ledcSetup(kBasePwmChannelReverse, robot_config::kBasePwmFrequencyHz, robot_config::kBasePwmResolutionBits);
    ledcAttachPin(board_pins::kBtsRpwm, kBasePwmChannelForward);
    ledcAttachPin(board_pins::kBtsLpwm, kBasePwmChannelReverse);
    ledcWrite(kBasePwmChannelForward, 0);
    ledcWrite(kBasePwmChannelReverse, 0);
}

void begin_reaction_wheel() {
    g_reaction_sensor.init(&Wire);
    g_reaction_driver.voltage_power_supply = robot_config::kReactionSupplyVoltage;
    g_reaction_driver.pwm_frequency = 25000;
    g_reaction_driver.init();

    g_reaction_motor.linkSensor(&g_reaction_sensor);
    g_reaction_motor.linkDriver(&g_reaction_driver);
    g_reaction_motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
    g_reaction_motor.controller = MotionControlType::torque;
    g_reaction_motor.torque_controller = TorqueControlType::voltage;
    g_reaction_motor.voltage_limit = fmaxf(
        robot_config::kReactionVoltageLimitHil,
        robot_config::kReactionVoltageLimitOnboard
    );
    g_reaction_motor.init();
    g_reaction_motor.initFOC();
}

float read_battery_voltage() {
    if (!robot_config::kBatterySenseEnabled) {
        return NAN;
    }
    const int raw = analogRead(board_pins::kBatterySenseAdc);
    const float sensed = (static_cast<float>(raw) / static_cast<float>(robot_config::kBatteryAdcMax)) *
                         robot_config::kBatteryAdcRefV;
    return sensed * robot_config::kBatteryDividerScale;
}

void poll_bridge_command() {
    if (!g_wifi_ready) {
        return;
    }

    int packet_size = g_udp.parsePacket();
    while (packet_size > 0) {
        char buf[192];
        const int n = g_udp.read(buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            StaticJsonDocument<192> doc;
            if (deserializeJson(doc, buf) == DeserializationError::Ok) {
                g_bridge_command.reaction_norm = clampf(doc["rt"] | 0.0f, -1.0f, 1.0f);
                g_bridge_command.base_norm = clampf(doc["dt"] | 0.0f, -1.0f, 1.0f);
                g_bridge_command.estop = (doc["estop"] | 0) != 0;
                g_bridge_command.seq = doc["seq"] | 0u;
                g_bridge_command.last_rx_us = micros();
                g_bridge_command.valid = true;
            }
        }
        packet_size = g_udp.parsePacket();
    }
}

bool safety_allows_motion(const AttitudeEstimate& attitude, uint32_t now_us) {
    imu_sample_t imu_sample{};
    imu_sample.pitch_rad = attitude.pitch_rad;
    imu_sample.roll_rad = attitude.roll_rad;
    imu_sample.pitch_rate_rad_s = attitude.pitch_rate_rad_s;
    imu_sample.roll_rate_rad_s = attitude.roll_rate_rad_s;
    imu_sample.timestamp_us = now_us;
    imu_sample.sequence = g_telem_seq;
    imu_sample.valid = attitude.valid;

    const uint32_t last_cmd_us = (robot_config::IsHilMode() && g_bridge_command.valid) ? g_bridge_command.last_rx_us : now_us;
    control_loop_safety_limits_t limits{};
    limits.cmd_timeout_us = robot_config::kCommandTimeoutUs;
    limits.tilt_trip_count = CTRL_TILT_TRIP_COUNT;
    limits.tilt_limit_rad = CTRL_CRASH_ANGLE_RAD;

    const bool safe = control_loop_safety_check(
        &g_fault_state,
        &g_safety_state,
        &limits,
        &imu_sample,
        now_us,
        last_cmd_us
    );
    if (!safe) {
        return false;
    }

    if (fabsf(attitude.pitch_rate_rad_s) > CTRL_MAX_PITCH_ROLL_RATE_RAD_S ||
        fabsf(attitude.roll_rate_rad_s) > CTRL_MAX_PITCH_ROLL_RATE_RAD_S) {
        fault_manager_latch(&g_fault_state, FAULT_RATE_LIMIT);
        return false;
    }

    const float battery_v = read_battery_voltage();
    if (!isnan(battery_v) && battery_v < robot_config::kBatteryUndervoltageV) {
        fault_manager_latch(&g_fault_state, FAULT_UNDERVOLTAGE);
        return false;
    }

    return !fault_manager_is_latched(&g_fault_state);
}

AppliedOutputs compute_hil_outputs(uint32_t now_us) {
    AppliedOutputs outputs;
    const bool timed_out = !g_bridge_command.valid ||
                           ((uint32_t)(now_us - g_bridge_command.last_rx_us) > robot_config::kCommandTimeoutUs);
    outputs.estop = timed_out || g_bridge_command.estop || fault_manager_is_latched(&g_fault_state);
    outputs.reaction_norm = outputs.estop ? 0.0f : g_bridge_command.reaction_norm;
    outputs.base_norm = outputs.estop || !robot_config::kBaseWheelEnabledInHil ? 0.0f : g_bridge_command.base_norm;
    return outputs;
}

AppliedOutputs compute_onboard_outputs(
    const AttitudeEstimate& attitude,
    const ReactionWheelSample& reaction,
    const BaseWheelEncoderSample& base_encoder
) {
    AppliedOutputs outputs;
    outputs.estop = fault_manager_is_latched(&g_fault_state) || !attitude.valid || !reaction.valid;
    if (outputs.estop) {
        memset(g_u_eff_applied, 0, sizeof(g_u_eff_applied));
        return outputs;
    }

    if (robot_config::kControlMode == robot_config::ControlMode::kOnboardExplicitSplit) {
        const float rw_norm = clampf(
            -(
                robot_config::kOnboardSplitRollKp * attitude.roll_rad +
                robot_config::kOnboardSplitRollKd * attitude.roll_rate_rad_s +
                robot_config::kOnboardSplitWheelDamp * reaction.speed_rad_s
            ),
            -1.0f,
            1.0f
        );
        float base_norm = -(
            robot_config::kOnboardSplitPitchKp * attitude.pitch_rad +
            robot_config::kOnboardSplitPitchKd * attitude.pitch_rate_rad_s
        );
        if (base_encoder.valid) {
            base_norm += -(
                robot_config::kOnboardSplitBaseHoldKp * base_encoder.pos_m +
                robot_config::kOnboardSplitBaseHoldKd * base_encoder.vel_m_s
            );
        }
        outputs.reaction_norm = rw_norm;
        outputs.base_norm = robot_config::kBaseWheelEnabledInOnboard ? clampf(base_norm, -1.0f, 1.0f) : 0.0f;
        outputs.estop = false;
        g_u_eff_applied[0] = outputs.reaction_norm * CTRL_WHEEL_TORQUE_LIMIT_NM;
        g_u_eff_applied[1] = outputs.base_norm * CTRL_BASE_FORCE_SOFT_LIMIT;
        g_u_eff_applied[2] = 0.0f;
        return outputs;
    }

    float y[CTRL_NY] = {0.0f};
    y[0] = attitude.pitch_rad;
    y[1] = attitude.roll_rad;
    y[2] = attitude.pitch_rate_rad_s;
    y[3] = attitude.roll_rate_rad_s;
    y[4] = reaction.speed_rad_s;
    if (robot_config::kHasBaseWheelEncoder && CTRL_NY >= 9) {
        y[5] = base_encoder.valid ? base_encoder.pos_m : 0.0f;
        y[6] = 0.0f;
        y[7] = base_encoder.valid ? base_encoder.vel_m_s : 0.0f;
        y[8] = 0.0f;
    }
    const float refs[2] = {CTRL_X_REF, CTRL_Y_REF};
    float u_cmd[3] = {0.0f, 0.0f, 0.0f};
    actuator_cmd_t guarded{};

    estimator_predict(&g_estimator_state, CTRL_A, CTRL_B, g_u_eff_applied);
    estimator_update(&g_estimator_state, &CTRL_C[0][0], &CTRL_L[0][0], y, CTRL_NY);

    controller_step(
        &g_controller_state,
        g_estimator_state.x,
        refs,
        CTRL_CONTROL_DT,
        CTRL_KI_BASE,
        CTRL_INT_CLAMP,
        CTRL_U_BLEED,
        CTRL_UPRIGHT_ANGLE_THRESH,
        CTRL_UPRIGHT_VEL_THRESH,
        CTRL_UPRIGHT_POS_THRESH,
        CTRL_K_DU,
        CTRL_MAX_DU,
        CTRL_MAX_U,
        u_cmd
    );

    actuator_guard_apply(
        reaction.speed_rad_s,
        0.0f,
        0.0f,
        u_cmd,
        robot_config::kBaseWheelEnabledInOnboard,
        CTRL_MAX_WHEEL_SPEED_RAD_S,
        CTRL_MAX_BASE_SPEED_M_S,
        CTRL_BASE_DERATE_START_FRAC,
        CTRL_BASE_FORCE_SOFT_LIMIT,
        CTRL_BASE_SPEED_SOFT_LIMIT_FRAC,
        CTRL_WHEEL_TORQUE_LIMIT_NM,
        CTRL_WHEEL_MOTOR_KV_RPM_PER_V,
        CTRL_WHEEL_MOTOR_RESISTANCE_OHM,
        CTRL_WHEEL_CURRENT_LIMIT_A,
        CTRL_BUS_VOLTAGE_V,
        CTRL_WHEEL_GEAR_RATIO,
        CTRL_DRIVE_EFFICIENCY,
        &guarded
    );

    outputs.reaction_norm = guarded.wheel_cmd_nm / fmaxf(CTRL_WHEEL_TORQUE_LIMIT_NM, 1e-6f);
    outputs.base_norm = robot_config::kBaseWheelEnabledInOnboard
        ? guarded.base_x_cmd / fmaxf(CTRL_BASE_FORCE_SOFT_LIMIT, 1e-6f)
        : 0.0f;
    outputs.reaction_norm = clampf(outputs.reaction_norm, -1.0f, 1.0f);
    outputs.base_norm = clampf(outputs.base_norm, -1.0f, 1.0f);
    outputs.estop = false;

    g_u_eff_applied[0] = guarded.wheel_cmd_nm;
    g_u_eff_applied[1] = guarded.base_x_cmd;
    g_u_eff_applied[2] = 0.0f;
    return outputs;
}

void apply_base_command(float base_norm, bool estop) {
    if (estop) {
        base_norm = 0.0f;
    }
    if (robot_config::kInvertBaseCommand) {
        base_norm = -base_norm;
    }

    const float pwm_limit = robot_config::IsHilMode()
        ? robot_config::kBasePwmLimitHil
        : robot_config::kBasePwmLimitOnboard;
    const int pwm = static_cast<int>(255.0f * fabsf(clampf(base_norm, -1.0f, 1.0f)) * pwm_limit);
    if (base_norm >= 0.0f) {
        ledcWrite(kBasePwmChannelForward, pwm);
        ledcWrite(kBasePwmChannelReverse, 0);
    } else {
        ledcWrite(kBasePwmChannelForward, 0);
        ledcWrite(kBasePwmChannelReverse, pwm);
    }
}

void apply_reaction_command(float reaction_norm, bool estop) {
    if (estop) {
        reaction_norm = 0.0f;
    }
    if (robot_config::kInvertReactionCommand) {
        reaction_norm = -reaction_norm;
    }

    const float voltage_limit = robot_config::IsHilMode()
        ? robot_config::kReactionHilNormToVoltage
        : robot_config::kReactionVoltageLimitOnboard;
    g_reaction_target_voltage = clampf(reaction_norm, -1.0f, 1.0f) * voltage_limit;
}

void apply_outputs(const AppliedOutputs& outputs) {
    const bool estop = outputs.estop || fault_manager_is_latched(&g_fault_state);
    apply_reaction_command(outputs.reaction_norm, estop);
    apply_base_command(outputs.base_norm, estop);
    g_applied_outputs = outputs;
    g_applied_outputs.estop = estop;
}

void send_telemetry(
    const RawImuSample& imu,
    const AttitudeEstimate& attitude,
    const ReactionWheelSample& reaction,
    const BaseWheelEncoderSample& base_encoder,
    uint32_t now_us
) {
    if (!g_wifi_ready || !robot_config::kSendTelemetry) {
        return;
    }

    StaticJsonDocument<512> doc;
    doc["mode"] = robot_config::ControlModeName(robot_config::kControlMode);
    doc["ax"] = imu.ax_m_s2;
    doc["ay"] = imu.ay_m_s2;
    doc["az"] = imu.az_m_s2;
    doc["gx"] = degrees(imu.gx_rad_s);
    doc["gy"] = degrees(imu.gy_rad_s);
    doc["gz"] = degrees(imu.gz_rad_s);
    doc["pitch_rad"] = attitude.pitch_rad;
    doc["roll_rad"] = attitude.roll_rad;
    doc["reaction_angle"] = degrees(reaction.angle_rad);
    doc["reaction_speed"] = degrees(reaction.speed_rad_s);
    doc["base_pos_m"] = base_encoder.pos_m;
    doc["base_vel_m_s"] = base_encoder.vel_m_s;
    doc["base_encoder_valid"] = base_encoder.valid ? 1 : 0;
    doc["rw_cmd"] = g_applied_outputs.reaction_norm;
    doc["base_cmd"] = g_applied_outputs.base_norm;
    doc["fault"] = static_cast<int>(g_fault_state.code);
    doc["latched"] = fault_manager_is_latched(&g_fault_state) ? 1 : 0;
    const float battery_v = read_battery_voltage();
    doc["battery_v"] = isnan(battery_v) ? -1.0f : battery_v;
    doc["ts"] = now_us;
    doc["seq"] = g_telem_seq++;

    char out[512];
    const size_t len = serializeJson(doc, out);
    g_udp.beginPacket(g_pc_ip, robot_config::kPcTelemetryPort);
    g_udp.write(reinterpret_cast<const uint8_t*>(out), len);
    g_udp.endPacket();
}

void service_serial() {
    while (Serial.available() > 0) {
        const int ch = Serial.read();
        if (ch == 'r' || ch == 'R') {
            clear_faults_and_states();
            Serial.println("Faults and controller state cleared.");
        } else if (ch == 'e' || ch == 'E') {
            fault_manager_latch(&g_fault_state, FAULT_CMD_TIMEOUT);
            Serial.println("Manual ESTOP latched.");
        }
    }
}

void print_debug(const AttitudeEstimate& attitude, const ReactionWheelSample& reaction) {
    const uint32_t now_ms = millis();
    if (now_ms - g_last_debug_ms < robot_config::kDebugPeriodMs) {
        return;
    }
    g_last_debug_ms = now_ms;

    Serial.print("mode=");
    Serial.print(robot_config::ControlModeName(robot_config::kControlMode));
    Serial.print(" pitch_deg=");
    Serial.print(degrees(attitude.pitch_rad), 2);
    Serial.print(" roll_deg=");
    Serial.print(degrees(attitude.roll_rad), 2);
    Serial.print(" rw_dps=");
    Serial.print(degrees(reaction.speed_rad_s), 1);
    Serial.print(" rw_cmd=");
    Serial.print(g_applied_outputs.reaction_norm, 3);
    Serial.print(" base_cmd=");
    Serial.print(g_applied_outputs.base_norm, 3);
    Serial.print(" fault=");
    Serial.print(static_cast<int>(g_fault_state.code));
    Serial.print(" estop=");
    Serial.println(g_applied_outputs.estop ? 1 : 0);
}

}  // namespace

void setup() {
    Serial.begin(115200);
    delay(200);
    pinMode(board_pins::kStatusLed, OUTPUT);
    digitalWrite(board_pins::kStatusLed, LOW);

    Wire.begin(board_pins::kI2cSda, board_pins::kI2cScl);
    Wire.setClock(400000);
    init_bmi088();
    begin_base_motor_driver();
    begin_base_encoder();
    begin_reaction_wheel();
    begin_wifi_if_needed();

    control_loop_init();
    clear_faults_and_states();

    g_last_tick_us = micros();
    g_last_debug_ms = millis();
    digitalWrite(board_pins::kStatusLed, HIGH);

    Serial.println("ESP32 sim-to-real scaffold ready.");
}

void loop() {
    g_reaction_motor.loopFOC();
    g_reaction_motor.move(g_reaction_target_voltage);

    service_serial();
    poll_bridge_command();

    const uint32_t now_us = micros();
    if ((uint32_t)(now_us - g_last_tick_us) < robot_config::kLoopPeriodUs) {
        return;
    }
    g_last_tick_us += robot_config::kLoopPeriodUs;

    control_loop_tick_250hz();

    const float dt_s = static_cast<float>(robot_config::kLoopPeriodUs) * 1.0e-6f;
    const RawImuSample imu = read_bmi088();
    const ReactionWheelSample reaction = read_reaction_wheel();
    const BaseWheelEncoderSample base_encoder = read_base_encoder(dt_s);
    const AttitudeEstimate attitude = update_attitude(imu, dt_s);

    AppliedOutputs outputs;
    if (!safety_allows_motion(attitude, now_us)) {
        outputs.estop = true;
    } else if (robot_config::IsHilMode()) {
        outputs = compute_hil_outputs(now_us);
    } else {
        outputs = compute_onboard_outputs(attitude, reaction, base_encoder);
    }

    apply_outputs(outputs);
    send_telemetry(imu, attitude, reaction, base_encoder, now_us);
    print_debug(attitude, reaction);
}
