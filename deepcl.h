#pragma once

#include <stddef.h>

typedef struct dcl_matf{
    size_t h, w;
    float* data;
} dcl_matf;

typedef struct dcl_matd{
    size_t h, w;
    double* data;
} dcl_matd;

//////////////////////////////////
//            MATRIX
//////////////////////////////////
typedef struct dcl_matf_operations{
    void(*map)(dcl_matf* A, float(*f)(float),dcl_matf* result);
    void(*add)(dcl_matf* A, dcl_matf* B, dcl_matf* result);
    void(*sub)(dcl_matf* A, dcl_matf* B, dcl_matf* result);
    void(*mul)(dcl_matf* A, dcl_matf* B, dcl_matf* result);
    void(*had)(dcl_matf* A, dcl_matf* B, dcl_matf* result);
} dcl_matf_operations;

//////////////////////////////////
//             DEEP
//////////////////////////////////
