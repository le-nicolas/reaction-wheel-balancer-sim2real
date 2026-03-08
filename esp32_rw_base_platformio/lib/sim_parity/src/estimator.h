#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x[9];
} estimator_state_t;

void estimator_reset(estimator_state_t* s);
void estimator_predict(estimator_state_t* s, const float a[9][9], const float b[9][3], const float u_eff[3]);
void estimator_update(estimator_state_t* s, const float* c, const float* l, const float* y, uint32_t ny);

#ifdef __cplusplus
}
#endif

#endif
