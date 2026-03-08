#ifndef ACTUATOR_GUARD_H
#define ACTUATOR_GUARD_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    float wheel_cmd_nm;
    float base_x_cmd;
    float base_y_cmd;
} actuator_cmd_t;

void actuator_guard_apply(
    float wheel_speed_rad_s,
    float base_x_speed_m_s,
    float base_y_speed_m_s,
    const float u_applied[3],
    bool allow_base_motion,
    float max_wheel_speed_rad_s,
    float max_base_speed_m_s,
    float base_derate_start_frac,
    float base_force_soft_limit,
    float base_speed_soft_limit_frac,
    float wheel_torque_limit_nm,
    float wheel_motor_kv_rpm_per_v,
    float wheel_motor_resistance_ohm,
    float wheel_current_limit_a,
    float bus_voltage_v,
    float wheel_gear_ratio,
    float drive_efficiency,
    actuator_cmd_t* out
);

#endif
