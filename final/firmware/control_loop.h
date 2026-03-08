#ifndef CONTROL_LOOP_H
#define CONTROL_LOOP_H

#include <stdbool.h>
#include <stdint.h>

#include "actuator_guard.h"
#include "controller.h"
#include "encoder_driver.h"
#include "estimator.h"
#include "fault_manager.h"
#include "imu_driver.h"

typedef struct {
    uint32_t tick_count;
    uint32_t deadline_misses;
    uint32_t last_tick_start_us;
    uint32_t last_tick_end_us;
    uint32_t command_timeout_events;
    uint32_t tilt_limit_events;
} control_loop_stats_t;

typedef struct {
    uint32_t cmd_timeout_us;
    uint32_t tilt_trip_count;
    float tilt_limit_rad;
} control_loop_safety_limits_t;

typedef struct {
    uint32_t tilt_over_limit_count;
} control_loop_safety_state_t;

void control_loop_init(void);
void control_loop_tick_250hz(void);
const control_loop_stats_t* control_loop_get_stats(void);
void control_loop_safety_reset(control_loop_safety_state_t* state);
bool control_loop_safety_check(
    fault_state_t* faults,
    control_loop_safety_state_t* state,
    const control_loop_safety_limits_t* limits,
    const imu_sample_t* imu,
    uint32_t now_us,
    uint32_t last_command_us
);

#endif
