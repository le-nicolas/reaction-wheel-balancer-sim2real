#include "encoder_driver.h"

bool encoder_driver_init(void) {
    return true;
}

bool encoder_driver_get_latest(encoder_sample_t* out) {
    if (!out) {
        return false;
    }
    out->valid = false;
    return false;
}
