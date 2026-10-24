/* Unit tests for cellmon on synthetic packs where the right answer is known exactly.
   Built with AddressSanitizer and UndefinedBehaviorSanitizer (see Makefile). */
#include "cellmon.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            failures++;                                               \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);             \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
        }                                                             \
    } while (0)

#define N 24
#define CAPACITY_AH 1.6f
#define R_CELL 0.10f

/* Deterministic noise: xorshift + Box-Muller, so failures reproduce exactly. */
static unsigned rng_state = 12345u;
static float uniform(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state >> 8) + 0.5f) / 16777216.0f;
}
static float gauss(float sd) { return sd * sqrtf(-2.0f * logf(uniform())) * cosf(6.2831853f * uniform()); }

typedef struct {
    float soc[N];
    float extra_r[N];   /* added resistance, ohm */
    float leak_a[N];    /* self-discharge current, A */
    float v_glitch[N];  /* added to the reading this step */
    float t_extra[N];   /* added temperature, degC */
    int stuck[N];
    float stuck_value[N];
} pack_t;

/* 1 Hz drive: 15 min rest, 60 min of 1 A with +/-1 A steps every 7 s, then rest. */
static float load_at(uint32_t k)
{
    if (k < 900 || k >= 4500) return 0.0f;
    return (k / 7) % 2 ? 2.0f : 0.5f;
}

static void make_sample(pack_t *p, uint32_t k, cm_sample_t *x, const cm_config_t *cfg)
{
    x->current = load_at(k);
    for (int i = 0; i < N; i++) {
        p->soc[i] -= (x->current + p->leak_a[i]) / (CAPACITY_AH * 3600.0f);
        float ocv = cfg->ocv_v[0] + (cfg->ocv_v[CM_OCV_POINTS - 1] - cfg->ocv_v[0]) * p->soc[i];
        float v = ocv - x->current * (R_CELL + p->extra_r[i]) + gauss(0.001f) + p->v_glitch[i];
        v = roundf(v * 1000.0f) / 1000.0f;
        if (p->stuck[i]) {
            if (p->stuck_value[i] == 0.0f) p->stuck_value[i] = v;
            v = p->stuck_value[i];
        }
        x->v[i] = v;
        x->t[i] = roundf((25.0f + 0.3f * (float)(i % 3) + p->t_extra[i] + gauss(0.1f)) * 10.0f) / 10.0f;
    }
}

static void fresh(pack_t *p)
{
    memset(p, 0, sizeof(*p));
    for (int i = 0; i < N; i++) p->soc[i] = 0.9f;
    rng_state = 12345u;
}

static cm_config_t config(void)
{
    cm_config_t cfg;
    cm_default_config(&cfg);
    cfg.capacity_ah = CAPACITY_AH;
    return cfg;
}

static cm_state_t st;  /* static: firmware would never put this on the stack either */

static void test_init_bounds(void)
{
    cm_config_t cfg = config();
    CHECK(cm_init(&st, &cfg, 2) != 0, "fewer than 3 cells has no meaningful median");
    CHECK(cm_init(&st, &cfg, CM_MAX_CELLS + 1) != 0, "more than CM_MAX_CELLS rejected");
    CHECK(cm_init(&st, &cfg, N) == 0, "valid size accepted");
    CHECK(cm_state_size() < 32 * 1024, "state is %u bytes", cm_state_size());
}

static uint32_t run(pack_t *p, uint32_t steps, void (*fault)(pack_t *, uint32_t))
{
    cm_config_t cfg = config();
    cm_sample_t x;
    cm_init(&st, &cfg, N);
    for (uint32_t k = 0; k < steps; k++) {
        if (fault) fault(p, k);
        make_sample(p, k, &x, &cfg);
        cm_step(&st, &x);
    }
    uint32_t alarmed = 0;
    for (int i = 0; i < N; i++) alarmed += st.cell[i].flags != 0;
    return alarmed;
}

static void test_healthy_pack_raises_nothing(void)
{
    pack_t p;
    fresh(&p);
    for (int i = 0; i < N; i++) p.extra_r[i] = 0.004f * (float)(i % 5);  /* normal spread */
    CHECK(run(&p, 5400, NULL) == 0, "healthy pack alarmed");
}

static void leak_cell_7(pack_t *p, uint32_t k) { p->leak_a[7] = k >= 2000 ? 0.05f : 0.0f; }

static void test_self_discharge_is_measured_during_rest(void)
{
    pack_t p;
    fresh(&p);
    run(&p, 5400, leak_cell_7);
    CHECK(st.cell[7].flags & CM_SELF_DISCHARGE, "50 mA leak not flagged (flags %u)", st.cell[7].flags);
    CHECK(fabsf(st.cell[7].leak_a - 0.05f) < 0.008f, "leak estimate %.4f A, injected 0.050", (double)st.cell[7].leak_a);
    for (int i = 0; i < N; i++)
        if (i != 7) CHECK(st.cell[i].flags == 0, "cell %d flagged %u", i, st.cell[i].flags);
}

static void resistive_cell_3(pack_t *p, uint32_t k) { p->extra_r[3] = k >= 1200 ? 0.06f : 0.0f; }

static void test_pulse_resistance_tracks_extra_resistance(void)
{
    pack_t p;
    fresh(&p);
    run(&p, 5400, resistive_cell_3);
    /* Test temperature is 25 degC, so the 25 degC-equivalent value equals the injected one. */
    CHECK(st.cell[3].flags & CM_HIGH_RESISTANCE, "60 mOhm extra not flagged");
    CHECK(fabsf(st.cell[3].pulse_r - 0.06f) < 0.006f, "pulse resistance %.4f, injected 0.060", (double)st.cell[3].pulse_r);
}

static void stuck_cell_11(pack_t *p, uint32_t k) { p->stuck[11] = k >= 1500; }

static void test_stuck_sensor(void)
{
    pack_t p;
    fresh(&p);
    run(&p, 3600, stuck_cell_11);
    CHECK(st.cell[11].flags & CM_SENSOR_STUCK, "stuck reading not flagged (flags %u)", st.cell[11].flags);
}

static void glitch_cell_5(pack_t *p, uint32_t k) { p->v_glitch[5] = (k == 4801 || k == 4802) ? 0.3f : 0.0f;  /* at rest: steady load */ }

static void test_spike_must_come_back(void)
{
    pack_t p;
    fresh(&p);
    run(&p, 5400, glitch_cell_5);
    CHECK(st.cell[5].flags & CM_SENSOR_SPIKE, "300 mV glitch not flagged");
}

static void hot_cell_9(pack_t *p, uint32_t k) { p->t_extra[9] = k >= 2000 ? fminf(6.0f, (float)(k - 2000) / 300.0f) : 0.0f; }

static void test_temperature_rise_over_peers(void)
{
    pack_t p;
    fresh(&p);
    run(&p, 5400, hot_cell_9);
    CHECK(st.cell[9].flags & CM_TEMP_RISE, "cell 6 degC above peers not flagged");
    /* Cells 1, 4, 7, ... read 0.3 degC high all along (thermistor offset): learned, not flagged. */
    CHECK(!(st.cell[1].flags & CM_TEMP_RISE), "constant sensor offset flagged");
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    struct { const char *name; void (*fn)(void); } tests[] = {
        {"init bounds and state size", test_init_bounds},
        {"healthy pack raises nothing", test_healthy_pack_raises_nothing},
        {"self-discharge measured during rest", test_self_discharge_is_measured_during_rest},
        {"pulse resistance tracks extra resistance", test_pulse_resistance_tracks_extra_resistance},
        {"stuck sensor", test_stuck_sensor},
        {"spike must come back", test_spike_must_come_back},
        {"temperature rise over peers", test_temperature_rise_over_peers},
    };
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        int before = failures;
        tests[i].fn();
        printf("%s %s\n", failures == before ? "PASS" : "FAIL", tests[i].name);
    }
    printf("%d failure(s)\n", failures);
    return failures != 0;
}
