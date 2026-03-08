#include "fault_manager.h"

void fault_manager_reset(fault_state_t* s) {
    if (!s) {
        return;
    }
    s->latched = false;
    s->code = FAULT_NONE;
    s->deadline_miss_count = 0;
}

void fault_manager_latch(fault_state_t* s, fault_code_t code) {
    if (!s || s->latched) {
        return;
    }
    s->latched = true;
    s->code = code;
}

bool fault_manager_is_latched(const fault_state_t* s) {
    return s && s->latched;
}
