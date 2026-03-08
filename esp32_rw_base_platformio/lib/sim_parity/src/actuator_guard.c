#include <math.h>

#include "actuator_guard.h"

static float clampf(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

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
) {
    float wheel_speed_abs;
    float kv_rad_per_s_per_v;
    float ke_v_per_rad_s;
    float motor_speed;
    float v_eff;
    float back_emf_v;
    float headroom_v;
    float i_voltage_limited;
    float i_available;
    float wheel_dynamic_limit;
    float wheel_limit;
    float wheel_cmd;
    float base_x_cmd;
    float base_y_cmd;
    float soft_speed;
    float span;
    float soft_scale;
    float base_derate_start;
    float base_margin;
    float bx_scale = 1.0f;
    float by_scale = 1.0f;
    if (!out || !u_applied) {
        return;
    }

    wheel_speed_abs = fabsf(wheel_speed_rad_s);
    kv_rad_per_s_per_v = wheel_motor_kv_rpm_per_v * (2.0f * 3.14159265358979323846f / 60.0f);
    if (kv_rad_per_s_per_v < 1e-9f) {
        kv_rad_per_s_per_v = 1e-9f;
    }
    ke_v_per_rad_s = 1.0f / kv_rad_per_s_per_v;
    motor_speed = wheel_speed_abs * fmaxf(wheel_gear_ratio, 1e-9f);
    v_eff = bus_voltage_v * clampf(drive_efficiency, 0.05f, 1.0f);
    back_emf_v = ke_v_per_rad_s * motor_speed;
    headroom_v = fmaxf(v_eff - back_emf_v, 0.0f);
    i_voltage_limited = headroom_v / fmaxf(wheel_motor_resistance_ohm, 1e-9f);
    i_available = fminf(wheel_current_limit_a, i_voltage_limited);
    wheel_dynamic_limit = ke_v_per_rad_s * i_available * wheel_gear_ratio * drive_efficiency;
    wheel_limit = fminf(wheel_torque_limit_nm, wheel_dynamic_limit);
    wheel_cmd = clampf(u_applied[0], -wheel_limit, wheel_limit);
    if (wheel_speed_abs >= max_wheel_speed_rad_s && wheel_cmd * wheel_speed_rad_s > 0.0f) {
        wheel_cmd = 0.0f;
    }

    base_derate_start = base_derate_start_frac * max_base_speed_m_s;
    if (fabsf(base_x_speed_m_s) > base_derate_start) {
        base_margin = fmaxf(max_base_speed_m_s - base_derate_start, 1e-6f);
        bx_scale = fmaxf(0.0f, 1.0f - (fabsf(base_x_speed_m_s) - base_derate_start) / base_margin);
    }
    if (fabsf(base_y_speed_m_s) > base_derate_start) {
        base_margin = fmaxf(max_base_speed_m_s - base_derate_start, 1e-6f);
        by_scale = fmaxf(0.0f, 1.0f - (fabsf(base_y_speed_m_s) - base_derate_start) / base_margin);
    }
    if (!allow_base_motion) {
        base_x_cmd = 0.0f;
        base_y_cmd = 0.0f;
    } else {
        base_x_cmd = u_applied[1] * bx_scale;
        base_y_cmd = u_applied[2] * by_scale;
    }

    soft_speed = clampf(base_speed_soft_limit_frac, 0.05f, 0.95f) * max_base_speed_m_s;
    if (fabsf(base_x_speed_m_s) > soft_speed && base_x_cmd * base_x_speed_m_s > 0.0f) {
        span = fmaxf(max_base_speed_m_s - soft_speed, 1e-6f);
        soft_scale = fmaxf(0.0f, 1.0f - (fabsf(base_x_speed_m_s) - soft_speed) / span);
        base_x_cmd *= soft_scale;
    }
    if (fabsf(base_y_speed_m_s) > soft_speed && base_y_cmd * base_y_speed_m_s > 0.0f) {
        span = fmaxf(max_base_speed_m_s - soft_speed, 1e-6f);
        soft_scale = fmaxf(0.0f, 1.0f - (fabsf(base_y_speed_m_s) - soft_speed) / span);
        base_y_cmd *= soft_scale;
    }

    base_x_cmd = clampf(base_x_cmd, -base_force_soft_limit, base_force_soft_limit);
    base_y_cmd = clampf(base_y_cmd, -base_force_soft_limit, base_force_soft_limit);
    if (fabsf(base_x_speed_m_s) >= max_base_speed_m_s && base_x_cmd * base_x_speed_m_s > 0.0f) {
        base_x_cmd = 0.0f;
    }
    if (fabsf(base_y_speed_m_s) >= max_base_speed_m_s && base_y_cmd * base_y_speed_m_s > 0.0f) {
        base_y_cmd = 0.0f;
    }

    out->wheel_cmd_nm = wheel_cmd;
    out->base_x_cmd = base_x_cmd;
    out->base_y_cmd = base_y_cmd;
}
