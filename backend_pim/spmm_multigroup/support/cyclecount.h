#include <inttypes.h>
#include <perfcounter.h>

// Timer
typedef struct perfcounter_cycles{
    perfcounter_t start[5];
    perfcounter_t end[5];
    perfcounter_t end2[5];
    uint64_t total[5];
} perfcounter_cycles;

void timer_init(perfcounter_cycles *cycles){
    cycles->total[0] = 0;
    cycles->total[1] = 0;
    cycles->total[2] = 0;
    cycles->total[3] = 0;
    cycles->total[4] = 0;
}

void timer_start(perfcounter_cycles *cycles, int i){
    cycles->start[i] = perfcounter_get(); // START TIMER
}

uint64_t timer_stop(perfcounter_cycles *cycles, int i){
    cycles->end[i] = perfcounter_get(); // STOP TIMER
    cycles->end2[i] = perfcounter_get(); // STOP TIMER
    cycles->total[i] += (((uint64_t)((uint64_t)(((cycles->end[i] >> 4) - (cycles->start[i] >> 4)) - ((cycles->end2[i] >> 4) - (cycles->end[i] >> 4))))) << 4);
    return cycles->total[i];
}

void timer_print(perfcounter_cycles *cycles, int i) { 
    printf("Cycles: %" PRIu64 "\t", cycles->total[i]);
}
