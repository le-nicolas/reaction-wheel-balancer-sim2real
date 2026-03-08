#ifndef ENCODER_DRIVER_H
#define ENCODER_DRIVER_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    float wheel_rate_rad_s;
    float base_x_pos_m;
    float base_y_pos_m;
    float base_x_rate_m_s;
    float base_y_rate_m_s;
    uint32_t timestamp_us;
    uint32_t sequence;
    bool valid;
} encoder_sample_t;

bool encoder_driver_init(void);
bool encoder_driver_get_latest(encoder_sample_t* out);

#endif
