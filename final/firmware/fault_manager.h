#ifndef FAULT_MANAGER_H
#define FAULT_MANAGER_H

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    FAULT_NONE = 0,
    FAULT_TILT_LIMIT,
    FAULT_RATE_LIMIT,
    FAULT_DEADLINE_MISS,
    FAULT_IMU_STALE,
    FAULT_CMD_TIMEOUT,
    FAULT_UNDERVOLTAGE,
    FAULT_OVERCURRENT,
} fault_code_t;

typedef struct {
    bool latched;
    fault_code_t code;
    uint32_t deadline_miss_count;
} fault_state_t;

void fault_manager_reset(fault_state_t* s);
void fault_manager_latch(fault_state_t* s, fault_code_t code);
bool fault_manager_is_latched(const fault_state_t* s);

#endif
