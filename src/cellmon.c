#include "cellmon.h"

#include <math.h>
#include <string.h>

void cm_default_config(cm_config_t *cfg)
{
    cfg->dt_s = 1.0f;
    cfg->rls_lambda = 0.9999f;      /* ~2.8 h memory at 1 Hz: absorbs aging, not a fault that grows in minutes */
    cfg->noise_alpha = 0.002f;
    cfg->min_sigma_v = 0.0015f;
    cfg->cusum_k = 5.0f;
    cfg->cusum_h = INFINITY;  /* off: see README, driving-time voltage residuals are too biased */
    cfg->temp_alpha = 0.0001f;  /* ~2.8 h: slower than a cell's ~20 min thermal time constant */
    cfg->min_sigma_t = 0.2f;
    cfg->temp_margin_c = 1.75f;
    cfg->temp_cusum_h = 300.0f;
    cfg->stuck_s = 120.0f;
    cfg->stuck_pack_mv = 20.0f;
    cfg->spike_v = 0.15f;
    cfg->max_extra_r_ohm = 0.025f;
    cfg->warmup_steps = 600;
    cfg->min_excitation = 3000.0f;  /* e.g. 25 min at 1.1 A */
    cfg->i_tau_s = 30.0f;
    cfg->r_nominal_ohm = 0.14f;
    for (int i = 0; i < CM_OCV_POINTS; i++)  /* placeholder linear table; callers load a measured one */
        cfg->ocv_v[i] = 3.0f + 1.2f * (float)i / (float)(CM_OCV_POINTS - 1);
    cfg->rest_current_a = 0.05f;
    cfg->rest_settle_s = 180.0f;
    cfg->rest_min_s = 300.0f;
    cfg->capacity_ah = 1.6f;
    cfg->leak_alarm_a = 0.015f;
    cfg->leak_z = 5.0f;
    cfg->r_activation_k = 3000.0f;
    cfg->pulse_min_di_a = 0.5f;
    cfg->pulse_alpha = 0.02f;
    cfg->pulse_min_events = 100;
    cfg->thermal_tau_s = 1200.0f;
    cfg->r_outlier_z = 5.0f;
}

/* k-th smallest of a[0..n-1] by Hoare selection; reorders a. O(n) expected, no recursion. */
static float select_kth(float *a, uint16_t n, uint16_t k)
{
    uint16_t lo = 0, hi = (uint16_t)(n - 1);
    while (lo < hi) {
        float pivot = a[(lo + hi) / 2];
        uint16_t i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;
            if (i <= j) {
                float tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
                i++;
                if (j == 0) break;
                j--;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else return a[k];
    }
    return a[k];
}

static float median(float *scratch, const float *x, uint16_t n)
{
    memcpy(scratch, x, n * sizeof(float));
    return select_kth(scratch, n, (uint16_t)(n / 2));
}

int cm_init(cm_state_t *s, const cm_config_t *cfg, uint16_t n_cells)
{
    if (n_cells < 3 || n_cells > CM_MAX_CELLS) return -1;
    memset(s, 0, sizeof(*s));
    s->cfg = *cfg;
    s->n = n_cells;
    for (uint16_t i = 0; i < n_cells; i++) {
        cm_cell_t *c = &s->cell[i];
        c->P[0][0] = 1.0f;
        c->P[1][1] = 1e-3f;
        c->P[2][2] = 1e-3f;
        c->P[3][3] = 1e-2f;
        c->var_v = cfg->min_sigma_v * cfg->min_sigma_v;
        c->var_t = cfg->min_sigma_t * cfg->min_sigma_t;
        c->t_P[0][0] = 1.0f;
        c->t_P[1][1] = 1.0f;
    }
    return 0;
}

/* Recursive least squares with forgetting. The update uses a clipped error so a fault in progress
   is not learned as the new normal at full speed. */
static void rls_update(cm_cell_t *c, const float x[CM_NREG], float e, float lambda)
{
    float px[CM_NREG];
    float denom = lambda;
    for (int i = 0; i < CM_NREG; i++) {
        px[i] = 0.0f;
        for (int j = 0; j < CM_NREG; j++) px[i] += c->P[i][j] * x[j];
        denom += x[i] * px[i];
    }
    float tr = 0.0f;
    for (int i = 0; i < CM_NREG; i++) {
        c->theta[i] += px[i] / denom * e;
        for (int j = 0; j < CM_NREG; j++) c->P[i][j] = (c->P[i][j] - px[i] * px[j] / denom) / lambda;
        tr += c->P[i][i];
    }
    /* Bound covariance growth during low excitation (long rests), a classic RLS windup guard. */
    if (tr > 1e4f)
        for (int i = 0; i < CM_NREG; i++)
            for (int j = 0; j < CM_NREG; j++) c->P[i][j] *= 1e4f / tr;
}

/* Slope of the OCV table (V per unit SOC) at a given open-circuit voltage, interpolated between
   node slopes so it changes smoothly. A stepwise slope makes the model's prediction jump when the
   pack crosses a table point, which looked like sensor spikes near empty where the curve is steep. */
static float ocv_slope(const cm_config_t *cfg, float ocv)
{
    const float step = 1.0f / (float)(CM_OCV_POINTS - 1);
    const float *v = cfg->ocv_v;
    int k = 0;
    while (k < CM_OCV_POINTS - 2 && ocv > v[k + 1]) k++;
    const float s0 = (k == 0) ? (v[1] - v[0]) / step : (v[k + 1] - v[k - 1]) / (2.0f * step);
    const float s1 = (k + 1 == CM_OCV_POINTS - 1) ? (v[k + 1] - v[k]) / step : (v[k + 2] - v[k]) / (2.0f * step);
    float w = (v[k + 1] > v[k]) ? (ocv - v[k]) / (v[k + 1] - v[k]) : 0.0f;
    w = fminf(1.0f, fmaxf(0.0f, w));
    return s0 + w * (s1 - s0);
}

static void cm_raise(cm_state_t *s, cm_cell_t *c, uint32_t flag, uint16_t *newly)
{
    if (!c->flags) {
        c->first_alarm_step = s->step;
        (*newly)++;
    }
    c->flags |= flag;
}

uint16_t cm_step(cm_state_t *s, const cm_sample_t *x)
{
    const cm_config_t *cfg = &s->cfg;
    const uint16_t n = s->n;
    const float med_v = median(s->scratch, x->v, n);
    const float med_t = median(s->scratch, x->t, n);
    s->excitation += x->current * x->current * cfg->dt_s;
    /* Checks that only compare with peers arm after a short warm-up. The voltage model also needs
       enough current history to have learned each cell's resistance before its residual means much. */
    const int armed = s->step >= cfg->warmup_steps;
    const int model_armed = armed && s->excitation >= cfg->min_excitation;
    /* Resistance rises roughly 2x from 25 to 0 degC; divide by this to compare at 25 degC. */
    const float to_25c = expf(cfg->r_activation_k * (1.0f / (med_t + 273.15f) - 1.0f / 298.15f));
    const float sat = 3.0f;
    uint16_t newly = 0;

    s->ah += x->current * cfg->dt_s / 3600.0f;
    s->i_filt += cfg->dt_s / cfg->i_tau_s * (x->current - s->i_filt);
    s->heat_filt += cfg->dt_s / cfg->thermal_tau_s * (x->current * x->current - s->heat_filt);
    const float slope = ocv_slope(cfg, med_v + x->current * cfg->r_nominal_ohm);
    const float reg[CM_NREG] = {1.0f, x->current, s->i_filt, s->ah * slope};

    /* Rest bookkeeping shared by all cells: time regressor for the self-discharge fit. */
    const int resting = fabsf(x->current) < cfg->rest_current_a;
    s->rest_steps = resting ? s->rest_steps + 1 : 0;
    const int settled = resting && (float)s->rest_steps * cfg->dt_s > cfg->rest_settle_s;
    float dt_mean = 0.0f;
    if (!settled) {
        s->rest_n = 0;
    } else {
        if (s->rest_n == 0) {
            s->rest_mean_t = 0.0f;
            s->rest_m2_t = 0.0f;
            s->rest_ocv_slope = slope;
            for (uint16_t i = 0; i < n; i++) {
                s->cell[i].rest_mean_r = 0.0f;
                s->cell[i].rest_c_tr = 0.0f;
                s->cell[i].rest_m2_r = 0.0f;
            }
        }
        s->rest_n++;
        const float t = (float)s->rest_n * cfg->dt_s;
        dt_mean = t - s->rest_mean_t;
        s->rest_mean_t += dt_mean / (float)s->rest_n;
        s->rest_m2_t += dt_mean * (t - s->rest_mean_t);
    }
    const int judge = settled && (float)s->rest_n * cfg->dt_s >= cfg->rest_min_s && armed;

    /* Resistance alarm level: the configured minimum, or further out if this pack's own cells are
       naturally more spread (robust spread from the median absolute deviation of pulse resistance). */
    float r_alarm = INFINITY;  /* no resistance alarms until the pack's spread is known */
    {
        uint16_t m = 0;
        for (uint16_t i = 0; i < n; i++)
            if (s->cell[i].pulse_events >= cfg->pulse_min_events) s->scratch[m++] = s->cell[i].pulse_r;
        if (m >= 3) {
            const float med_r = select_kth(s->scratch, m, (uint16_t)(m / 2));
            for (uint16_t i = 0; i < m; i++) s->scratch[i] = fabsf(s->scratch[i] - med_r);
            const float spread = 1.4826f * select_kth(s->scratch, m, (uint16_t)(m / 2));
            r_alarm = fmaxf(cfg->max_extra_r_ohm, med_r + cfg->r_outlier_z * spread);
        }
    }

    for (uint16_t i = 0; i < n; i++) {
        cm_cell_t *c = &s->cell[i];

        /* Voltage: residual against the pack median, minus this cell's learned normal offset. */
        const float r = x->v[i] - med_v;
        float pred = 0.0f;
        for (int j = 0; j < CM_NREG; j++) pred += c->theta[j] * reg[j];
        const float e = r - pred;
        float sigma = sqrtf(c->var_v);
        if (sigma < cfg->min_sigma_v) sigma = cfg->min_sigma_v;
        const float z = e / sigma;

        /* Pulse resistance: across a sharp current step, everything slow (state of charge, capacity,
           polarization) cancels, and the extra voltage step of this cell over the median is its
           resistance difference times the current step. */
        const float d_i = x->current - s->last_current;
        if (s->step > 0 && fabsf(d_i) >= cfg->pulse_min_di_a) {
            const float r_est = -(r - c->last_r) / d_i / to_25c;
            /* Plain running mean until the estimate is trusted, then an EWMA that can follow aging. */
            c->pulse_r += (c->pulse_events < cfg->pulse_min_events ? 1.0f / (float)(c->pulse_events + 1) : cfg->pulse_alpha) *
                          (r_est - c->pulse_r);
            c->pulse_events++;
        }

        /* A glitch is a jump in the model error that comes straight back, while the load is steady
           (a current step moves every cell's voltage, by more in a higher-resistance cell). */
        const float jump = e - c->last_err;
        if (fabsf(d_i) >= cfg->pulse_min_di_a) {
            c->spike_window = 0;
        } else if (c->spike_window > 0) {
            c->spike_window--;
            if (armed && jump * c->spike_sign < -0.5f * cfg->spike_v) {
                cm_raise(s, c, CM_SENSOR_SPIKE, &newly);
                c->spike_window = 0;
            }
        } else if (fabsf(jump) > cfg->spike_v) {
            c->spike_window = 5;
            c->spike_sign = jump > 0.0f ? 1.0f : -1.0f;
        }

        if (model_armed) {
            c->cusum_v = fmaxf(0.0f, c->cusum_v - z - cfg->cusum_k);
            if (c->cusum_v > cfg->cusum_h) cm_raise(s, c, CM_VOLT_DRIFT, &newly);
        }
        if (armed && c->pulse_events >= cfg->pulse_min_events && c->pulse_r > r_alarm)
            cm_raise(s, c, CM_HIGH_RESISTANCE, &newly);
        const float e_clip = fmaxf(-sat * sigma, fminf(sat * sigma, e));
        /* Do not learn the noise level from samples that already look anomalous; otherwise sigma
           grows with the fault and the fault never looks unusual. */
        if (!model_armed || fabsf(z) < sat) c->var_v += cfg->noise_alpha * (e_clip * e_clip - c->var_v);
        rls_update(c, reg, e_clip, cfg->rls_lambda);

        if (settled) {
            const float dr = r - c->rest_mean_r;
            c->rest_mean_r += dr / (float)s->rest_n;
            c->rest_c_tr += dt_mean * (r - c->rest_mean_r);
            c->rest_m2_r += dr * (r - c->rest_mean_r);
            if (judge && s->rest_m2_t > 0.0f) {
                const float slope_v = c->rest_c_tr / s->rest_m2_t;  /* V per second */
                const float resid_var = (c->rest_m2_r - slope_v * c->rest_c_tr) / (float)(s->rest_n - 2);
                const float slope_se = sqrtf(fmaxf(resid_var, 0.0f) / s->rest_m2_t);
                const float to_amps = -cfg->capacity_ah * 3600.0f / s->rest_ocv_slope;
                c->leak_a = slope_v * to_amps;
                const float leak_se = slope_se * fabsf(to_amps);
                if (c->leak_a > cfg->leak_alarm_a && c->leak_a > cfg->leak_z * leak_se)
                    cm_raise(s, c, CM_SELF_DISCHARGE, &newly);
            }
        }

        /* Temperature: hotter than the pack median by more than this cell's own heating explains.
           Model: rt ~ a + b * H, where H is pack heating (I^2) filtered with the cell's thermal time
           constant; b absorbs normal differences in resistance and cooling from cell to cell. */
        const float rt = x->t[i] - med_t;
        if (s->step == 0) c->t_theta[0] = rt;
        const float et = rt - (c->t_theta[0] + c->t_theta[1] * s->heat_filt);
        /* Sensor noise from sample-to-sample differences: a cell warming slowly cannot inflate it,
           which a variance of the residual itself would (the fault would then never look unusual). */
        if (s->step > 0) {
            const float d = rt - c->last_rt;
            c->var_t += cfg->noise_alpha * (0.5f * d * d - c->var_t);
        }
        c->last_rt = rt;
        float sigma_t = sqrtf(c->var_t);
        if (sigma_t < cfg->min_sigma_t) sigma_t = cfg->min_sigma_t;
        /* CUSUM on the excess over a physical margin: normal self-heating differences stay inside it. */
        if (armed) {
            c->cusum_t = fmaxf(0.0f, c->cusum_t + (et - cfg->temp_margin_c) / sigma_t);
            if (c->cusum_t > cfg->temp_cusum_h) cm_raise(s, c, CM_TEMP_RISE, &newly);
        }
        if (!armed || et < cfg->temp_margin_c) {
            /* 2-parameter RLS with slow forgetting, frozen while the cell is over the margin. */
            const float h[2] = {1.0f, s->heat_filt};
            const float ph0 = c->t_P[0][0] * h[0] + c->t_P[0][1] * h[1];
            const float ph1 = c->t_P[1][0] * h[0] + c->t_P[1][1] * h[1];
            const float lam = 1.0f - cfg->temp_alpha;
            const float den = lam + h[0] * ph0 + h[1] * ph1;
            c->t_theta[0] += ph0 / den * et;
            c->t_theta[1] += ph1 / den * et;
            c->t_P[0][0] = (c->t_P[0][0] - ph0 * ph0 / den) / lam;
            c->t_P[0][1] = (c->t_P[0][1] - ph0 * ph1 / den) / lam;
            c->t_P[1][0] = c->t_P[0][1];
            c->t_P[1][1] = (c->t_P[1][1] - ph1 * ph1 / den) / lam;
        }

        /* Stuck sensor: the reading has not changed while the rest of the pack has. */
        const float med_mv = med_v * 1000.0f;
        if (s->step > 0 && x->v[i] == c->last_v) {
            c->stuck_steps++;
            if (armed && c->stuck_steps * cfg->dt_s >= cfg->stuck_s &&
                fabsf(med_mv - c->stuck_anchor_mv) >= cfg->stuck_pack_mv)
                cm_raise(s, c, CM_SENSOR_STUCK, &newly);
        } else {
            c->stuck_steps = 0;
            c->stuck_anchor_mv = med_mv;
        }
        c->last_v = x->v[i];
        c->last_err = e;
        c->last_r = r;
    }
    s->last_current = x->current;
    s->step++;
    return newly;
}

void cm_run_ex(const cm_config_t *cfg, uint16_t n, uint32_t steps, const float *v, const float *t,
               const float *current, uint32_t *first_alarm, uint32_t *flags, float *pulse_r, float *leak_a)
{
    static cm_state_t s; /* static, not stack: the state is ~24 KB at 128 cells */
    static cm_sample_t x;
    cm_init(&s, cfg, n);
    for (uint32_t k = 0; k < steps; k++) {
        memcpy(x.v, v + (uint64_t)k * n, n * sizeof(float));
        memcpy(x.t, t + (uint64_t)k * n, n * sizeof(float));
        x.current = current[k];
        cm_step(&s, &x);
    }
    for (uint16_t i = 0; i < n; i++) {
        flags[i] = s.cell[i].flags;
        first_alarm[i] = s.cell[i].flags ? s.cell[i].first_alarm_step : UINT32_MAX;
        if (pulse_r) pulse_r[i] = s.cell[i].pulse_r;
        if (leak_a) leak_a[i] = s.cell[i].leak_a;
    }
}

void cm_run(const cm_config_t *cfg, uint16_t n, uint32_t steps, const float *v, const float *t,
            const float *current, uint32_t *first_alarm, uint32_t *flags)
{
    cm_run_ex(cfg, n, steps, v, t, current, first_alarm, flags, NULL, NULL);
}

uint32_t cm_state_size(void) { return (uint32_t)sizeof(cm_state_t); }
uint32_t cm_config_size(void) { return (uint32_t)sizeof(cm_config_t); }
