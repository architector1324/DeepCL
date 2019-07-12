#pragma once

#include <stddef.h>


//////////////////////////////////
//            MATRIX
//////////////////////////////////
typedef struct dcl_matf{
    size_t h, w;
    float* data;
} dcl_matf;

typedef struct dcl_matd{
    size_t h, w;
    double* data;
} dcl_matd;


typedef struct dcl_matf_operations{
    void(*map)(dcl_matf* A, float(*f)(float),dcl_matf* result);
    void(*add)(dcl_matf* A, dcl_matf* B, dcl_matf* result);
    void(*sub)(dcl_matf* A, dcl_matf* B, dcl_matf* result);
    void(*mul)(dcl_matf* A, dcl_matf* B, dcl_matf* result);
    void(*had)(dcl_matf* A, dcl_matf* B, dcl_matf* result);
} dcl_matf_operations;

typedef struct dcl_matd_operations{
    void(*map)(dcl_matd* A, float(*f)(float),dcl_matd* result);
    void(*add)(dcl_matd* A, dcl_matd* B, dcl_matd* result);
    void(*sub)(dcl_matd* A, dcl_matd* B, dcl_matd* result);
    void(*mul)(dcl_matd* A, dcl_matd* B, dcl_matd* result);
    void(*had)(dcl_matd* A, dcl_matd* B, dcl_matd* result);
} dcl_matd_operations;


//////////////////////////////////
//             DEEP
//////////////////////////////////
