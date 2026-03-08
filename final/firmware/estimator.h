#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <stdint.h>

typedef struct {
    float x[9];
} estimator_state_t;

void estimator_reset(estimator_state_t* s);
void estimator_predict(estimator_state_t* s, const float a[9][9], const float b[9][3], const float u_eff[3]);
void estimator_update(estimator_state_t* s, const float c[5][9], const float l[9][5], const float y[5]);

#endif
