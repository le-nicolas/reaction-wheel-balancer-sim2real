#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <stdint.h>

typedef struct {
    float u_prev[3];
    float base_int[2];
} controller_state_t;

void controller_reset(controller_state_t* s);
void controller_step(
    controller_state_t* s,
    const float x_est[9],
    const float x_ref_y_ref[2],
    float control_dt,
    float ki_base,
    float int_clamp,
    float u_bleed,
    float upright_angle_thresh,
    float upright_vel_thresh,
    float upright_pos_thresh,
    const float k_du[3][12],
    const float max_du[3],
    const float max_u[3],
    float out_u_cmd[3]
);

#endif
