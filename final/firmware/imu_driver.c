#include "imu_driver.h"

bool imu_driver_init(void) {
    return true;
}

bool imu_driver_get_latest(imu_sample_t* out) {
    if (!out) {
        return false;
    }
    out->valid = false;
    return false;
}
