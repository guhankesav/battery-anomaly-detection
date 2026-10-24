/*
 * cellmon: real-time cell anomaly detection for a series battery pack.
 *
 * Designed like firmware: no heap, no recursion, fixed-size state, one call per sample.
 * Each cell is compared with the pack median, so pack-wide effects (load, temperature, state of
 * charge) cancel; what is left is modeled per cell and watched with CUSUM detectors.
 */
#ifndef CELLMON_H
#define CELLMON_H

#include <stdint.h>

#define CM_MAX_CELLS 128
#define CM_OCV_POINTS 21
#define CM_NREG 4

typedef struct {
    float v[CM_MAX_CELLS];  /* cell voltages, V */
    float t[CM_MAX_CELLS];  /* cell temperatures, degC */
    float current;          /* pack current, A, positive = discharge */
} cm_sample_t;

typedef struct {
    float dt_s;             /* sample period */
    float rls_lambda;       /* forgetting factor of the per-cell residual model */
    float noise_alpha;      /* EWMA weight for residual variance */
    float min_sigma_v;      /* floor on voltage residual sigma, V */
    float cusum_k;          /* CUSUM allowance, in sigma */
    float cusum_h;          /* CUSUM alarm threshold, in sigma */
    float temp_alpha;       /* EWMA weight for each cell's normal temperature offset */
    float min_sigma_t;      /* floor on temperature residual sigma, degC */
    float temp_margin_c;    /* degC above this cell's usual offset that is still normal */
    float temp_cusum_h;
    float stuck_s;          /* reading unchanged this long while the pack moves = stuck sensor */
    float stuck_pack_mv;    /* ...and the pack median moved at least this much */
    float spike_v;          /* step in a cell's residual larger than this = sensor spike */
    float max_extra_r_ohm;  /* resistance above the pack median, at 25 degC, that raises an alarm */
    uint32_t warmup_steps;  /* learn offsets before alarming */
    float min_excitation;   /* ...and until sum(I^2 dt) reaches this (A^2 s), so resistance is learned */
    float i_tau_s;          /* time constant of the filtered current (cell polarization) */
    float r_nominal_ohm;    /* typical cell resistance, to estimate open-circuit voltage under load */
    float ocv_v[CM_OCV_POINTS];  /* open-circuit voltage at SOC 0, 5, ..., 100 % (the BMS's OCV table) */
    /* Self-discharge during rest: with no current, only a leak (or a drifting sensor) moves a cell
       against its peers, so the slope of its residual over time measures leakage current. */
    float rest_current_a;   /* |I| below this counts as rest */
    float rest_settle_s;    /* ignore the first part of a rest while polarization relaxes */
    float rest_min_s;       /* minimum settled rest before judging */
    float capacity_ah;      /* nominal cell capacity, to turn a voltage slope into a current */
    float leak_alarm_a;     /* leakage that raises an alarm */
    float leak_z;           /* ...and the estimate must be this many standard errors above zero */
    float r_activation_k;   /* Arrhenius constant: resistance is compared at its 25 degC equivalent */
    float pulse_min_di_a;   /* current step large enough to measure resistance from */
    float pulse_alpha;      /* EWMA weight of each pulse-resistance measurement */
    uint32_t pulse_min_events;
    float thermal_tau_s;    /* cell thermal time constant, for the filtered heating regressor */
    float r_outlier_z;      /* ...and a cell must sit this many robust spreads above its pack's median */
} cm_config_t;

enum {
    CM_VOLT_DRIFT      = 1u << 0,  /* cell sags against its peers over time (e.g. internal short) */
    CM_TEMP_RISE       = 1u << 1,  /* cell runs hotter than its peers and its own history */
    CM_SENSOR_STUCK    = 1u << 2,
    CM_SENSOR_SPIKE    = 1u << 3,
    CM_HIGH_RESISTANCE = 1u << 4,  /* voltage drop under load larger than peers (connection, aging) */
    CM_SELF_DISCHARGE  = 1u << 5,  /* cell drains against its peers while the pack rests */
};

typedef struct {
    /* residual ~ theta0 + theta1 * I + theta2 * I_filtered + theta3 * Ah * dOCV/dSOC
       offset, ohmic resistance, polarization, and capacity mismatch (which moves a cell away from
       its peers in proportion to charge drawn, scaled by how steep the OCV curve is there). */
    float theta[CM_NREG];
    float P[CM_NREG][CM_NREG];
    float var_v;
    float cusum_v;
    float t_theta[2];       /* temperature residual ~ t_theta0 + t_theta1 * filtered I^2 */
    float t_P[2][2];
    float var_t;
    float cusum_t;
    float last_rt;
    float last_v;
    float last_err;
    float last_r;
    float pulse_r;          /* resistance above the pack median at 25 degC, from current steps */
    uint32_t pulse_events;
    float spike_sign;
    uint32_t spike_window;  /* samples left to see a jump reverse */
    float stuck_anchor_mv;  /* pack median when this cell's reading last changed */
    uint32_t stuck_steps;
    float rest_mean_r;      /* running regression of residual on rest time (Welford form) */
    float rest_c_tr;
    float rest_m2_r;
    float leak_a;           /* latest self-discharge estimate */
    uint32_t flags;         /* latched CM_* bits */
    uint32_t first_alarm_step;
} cm_cell_t;

typedef struct {
    cm_config_t cfg;
    uint16_t n;
    uint32_t step;
    float ah;               /* charge throughput since start, Ah */
    float excitation;       /* sum of I^2 dt so far */
    float i_filt;
    float last_current;
    float heat_filt;
    uint32_t rest_steps;    /* consecutive rest samples */
    uint32_t rest_n;        /* settled rest samples in the regression */
    float rest_mean_t;
    float rest_m2_t;
    float rest_ocv_slope;   /* V per unit SOC where the pack is resting */
    float scratch[CM_MAX_CELLS];
    cm_cell_t cell[CM_MAX_CELLS];
} cm_state_t;

void cm_default_config(cm_config_t *cfg);
int cm_init(cm_state_t *s, const cm_config_t *cfg, uint16_t n_cells);
/* Processes one sample; returns the number of cells with a newly raised alarm. */
uint16_t cm_step(cm_state_t *s, const cm_sample_t *x);
/* Runs a whole recording (row-major v and t, [steps][n]); writes each cell's first alarm step and
   flags. Convenience entry point for the Python harness. */
void cm_run(const cm_config_t *cfg, uint16_t n, uint32_t steps, const float *v, const float *t,
            const float *current, uint32_t *first_alarm, uint32_t *flags);
/* Same, also returning each cell's final pulse-resistance and leakage estimates (may be NULL). */
void cm_run_ex(const cm_config_t *cfg, uint16_t n, uint32_t steps, const float *v, const float *t,
               const float *current, uint32_t *first_alarm, uint32_t *flags, float *pulse_r, float *leak_a);
uint32_t cm_state_size(void);
uint32_t cm_config_size(void);  /* lets bindings verify their struct layout */

#endif
