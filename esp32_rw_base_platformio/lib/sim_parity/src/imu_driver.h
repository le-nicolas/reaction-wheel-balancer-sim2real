#ifndef IMU_DRIVER_H
#define IMU_DRIVER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float pitch_rad;
    float roll_rad;
    float pitch_rate_rad_s;
    float roll_rate_rad_s;
    uint32_t timestamp_us;
    uint32_t sequence;
    bool valid;
} imu_sample_t;

bool imu_driver_init(void);
bool imu_driver_get_latest(imu_sample_t* out);

#ifdef __cplusplus
}
#endif

#endif
