#include <string.h>
#include <math.h>

#include "control_loop.h"

static control_loop_stats_t g_stats;

void control_loop_init(void) {
    memset(&g_stats, 0, sizeof(g_stats));
}

void control_loop_tick_250hz(void) {
    g_stats.tick_count++;
}

const control_loop_stats_t* control_loop_get_stats(void) {
    return &g_stats;
}

void control_loop_safety_reset(control_loop_safety_state_t* state) {
    if (!state) {
        return;
    }
    memset(state, 0, sizeof(*state));
}

bool control_loop_safety_check(
    fault_state_t* faults,
    control_loop_safety_state_t* state,
    const control_loop_safety_limits_t* limits,
    const imu_sample_t* imu,
    uint32_t now_us,
    uint32_t last_command_us
) {
    uint32_t dt_us;
    if (!faults || !state || !limits || !imu) {
        return false;
    }
    if (fault_manager_is_latched(faults)) {
        return false;
    }
    if (!imu->valid) {
        fault_manager_latch(faults, FAULT_IMU_STALE);
        return false;
    }

    dt_us = now_us - last_command_us;
    if (dt_us > limits->cmd_timeout_us) {
        g_stats.command_timeout_events++;
        fault_manager_latch(faults, FAULT_CMD_TIMEOUT);
        return false;
    }

    if (fabsf(imu->pitch_rad) > limits->tilt_limit_rad || fabsf(imu->roll_rad) > limits->tilt_limit_rad) {
        state->tilt_over_limit_count++;
    } else {
        state->tilt_over_limit_count = 0;
    }
    if (state->tilt_over_limit_count >= limits->tilt_trip_count) {
        g_stats.tilt_limit_events++;
        fault_manager_latch(faults, FAULT_TILT_LIMIT);
        return false;
    }

    return true;
}
