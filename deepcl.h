#pragma once
#include <stddef.h>
#include <stdlib.h>


//////////////////////////////////
//            MATRIX
//////////////////////////////////
typedef struct dcl_matf{
    size_t h, w;
    float* data;
} dcl_matf;

typedef struct dcl_matf_operations{
    void(*map)(const dcl_matf* A, float(*f)(float), dcl_matf* result);
    void(*add)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
    void(*sub)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
    void(*mul)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
    void(*had)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
} dcl_matf_operations;



//////////////////////////////////
//             DEEP
//////////////////////////////////
// layer
typedef struct{
    dcl_matf* core;

    float(*activation)(float);
    float(*derivative)(float);
} dcl_layerf;


// API
void dcl_queryf(const dcl_matf* in, dcl_matf* preout, dcl_matf* out, const dcl_layerf* layer, const dcl_matf_operations* ops){
    if(layer->core && preout){
        ops->mul(layer->core, in, preout);
        ops->map(preout, layer->activation, out);
    } else 
        ops->map(in, layer->activation, out);
}