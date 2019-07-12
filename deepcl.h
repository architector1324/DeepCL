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
    void(*map)(const dcl_matf* A, float(*f)(float), dcl_matf* result);
    void(*add)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
    void(*sub)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
    void(*mul)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
    void(*had)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
} dcl_matf_operations;

typedef struct dcl_matd_operations{
    void(*map)(const dcl_matd* A, float(*f)(float),dcl_matd* result);
    void(*add)(const dcl_matd* A, const dcl_matd* B, dcl_matd* result);
    void(*sub)(const dcl_matd* A, const dcl_matd* B, dcl_matd* result);
    void(*mul)(const dcl_matd* A, const dcl_matd* B, dcl_matd* result);
    void(*had)(const dcl_matd* A, const dcl_matd* B, dcl_matd* result);
} dcl_matd_operations;


//////////////////////////////////
//             DEEP
//////////////////////////////////
