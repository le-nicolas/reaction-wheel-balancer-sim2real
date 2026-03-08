#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "controller.h"

static float clampf(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

void controller_reset(controller_state_t* s) {
    if (!s) {
        return;
    }
    memset(s, 0, sizeof(*s));
}

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
) {
    int i;
    int j;
    float x_ctrl[9];
    float z[12];
    float du_cmd[3];
    float du[3];
    float u_unc[3];
    bool near_upright;
    if (!s || !x_est || !x_ref_y_ref || !k_du || !max_du || !max_u || !out_u_cmd) {
        return;
    }

    memcpy(x_ctrl, x_est, sizeof(x_ctrl));
    x_ctrl[5] -= x_ref_y_ref[0];
    x_ctrl[6] -= x_ref_y_ref[1];

    s->base_int[0] = clampf(s->base_int[0] + x_ctrl[5] * control_dt, -int_clamp, int_clamp);
    s->base_int[1] = clampf(s->base_int[1] + x_ctrl[6] * control_dt, -int_clamp, int_clamp);

    for (i = 0; i < 9; ++i) {
        z[i] = x_ctrl[i];
    }
    for (i = 0; i < 3; ++i) {
        z[9 + i] = s->u_prev[i];
    }

    for (i = 0; i < 3; ++i) {
        float acc = 0.0f;
        for (j = 0; j < 12; ++j) {
            acc += k_du[i][j] * z[j];
        }
        du_cmd[i] = -acc;
    }
    du_cmd[1] += -ki_base * s->base_int[0];
    du_cmd[2] += -ki_base * s->base_int[1];

    for (i = 0; i < 3; ++i) {
        du[i] = clampf(du_cmd[i], -max_du[i], max_du[i]);
        u_unc[i] = s->u_prev[i] + du[i];
        out_u_cmd[i] = clampf(u_unc[i], -max_u[i], max_u[i]);
    }

    near_upright = fabsf(x_ctrl[0]) < upright_angle_thresh &&
                   fabsf(x_ctrl[1]) < upright_angle_thresh &&
                   fabsf(x_ctrl[2]) < upright_vel_thresh &&
                   fabsf(x_ctrl[3]) < upright_vel_thresh &&
                   fabsf(x_ctrl[5]) < upright_pos_thresh &&
                   fabsf(x_ctrl[6]) < upright_pos_thresh;
    if (near_upright) {
        for (i = 0; i < 3; ++i) {
            out_u_cmd[i] *= u_bleed;
            if (fabsf(out_u_cmd[i]) < 1e-3f) {
                out_u_cmd[i] = 0.0f;
            }
        }
    }

    for (i = 0; i < 3; ++i) {
        s->u_prev[i] = out_u_cmd[i];
    }
}
