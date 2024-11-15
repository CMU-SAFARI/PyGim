#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <getopt.h>
#include <unistd.h>
#include <string.h>

typedef struct Params {
    char* fileName;
    unsigned int dense_size;
    unsigned int nthreads;
} Params;


void usage() {
    fprintf(stderr,
            "\nUsage:  ./program [options]"
            "\n"
            "\nGeneral options:"
            "\n    -h        help"
            "\n"
            "\nBenchmark-specific options:"
            "\n    -f <F>    Input matrix file name (default=data/bcsstk01.mtx"
            "\n    -d <D>    # of columns in dense matrix (default=256)"
            "\n    -n <N>    # of threads (default=2)"
            "\n");
}


struct Params input_params(int argc, char **argv) {
    struct Params p;
    p.fileName      = "bcsstk18.mtx";
    p.dense_size    = 256;
    p.nthreads      = 2;

    int opt;
    while((opt = getopt(argc, argv, "hd:f:n:e:")) >= 0) {
        switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': p.dense_size    = atoi(optarg); break;
            case 'f': p.fileName      = optarg; break;
            case 'n': p.nthreads      = atoi(optarg); break;
            default:
                      fprintf(stderr, "\nUnrecognized option!\n");
                      usage();
                      exit(0);
        }
    }
    assert(NR_DPUS > 0 && "Invalid # of dpus!");

    return p;
}


#endif
