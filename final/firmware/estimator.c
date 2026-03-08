#include <string.h>

#include "estimator.h"

void estimator_reset(estimator_state_t* s) {
    if (!s) {
        return;
    }
    memset(s->x, 0, sizeof(s->x));
}

void estimator_predict(estimator_state_t* s, const float a[9][9], const float b[9][3], const float u_eff[3]) {
    int i;
    int j;
    float next_x[9];
    if (!s || !a || !b || !u_eff) {
        return;
    }
    for (i = 0; i < 9; ++i) {
        float acc = 0.0f;
        for (j = 0; j < 9; ++j) {
            acc += a[i][j] * s->x[j];
        }
        for (j = 0; j < 3; ++j) {
            acc += b[i][j] * u_eff[j];
        }
        next_x[i] = acc;
    }
    memcpy(s->x, next_x, sizeof(next_x));
}

void estimator_update(estimator_state_t* s, const float c[5][9], const float l[9][5], const float y[5]) {
    int i;
    int j;
    float cx[5];
    float innov[5];
    if (!s || !c || !l || !y) {
        return;
    }
    for (i = 0; i < 5; ++i) {
        float acc = 0.0f;
        for (j = 0; j < 9; ++j) {
            acc += c[i][j] * s->x[j];
        }
        cx[i] = acc;
        innov[i] = y[i] - cx[i];
    }
    for (i = 0; i < 9; ++i) {
        float corr = 0.0f;
        for (j = 0; j < 5; ++j) {
            corr += l[i][j] * innov[j];
        }
        s->x[i] += corr;
    }
}
