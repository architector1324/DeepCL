#pragma once
#include <stddef.h>
#include <stdlib.h>


/* 
    You can use this library on any device (include embedded systems), 
    no memory allocations at all!
*/ 

//////////////////////////////////
//            MATRIX
//////////////////////////////////
typedef struct dcl_matf{
    size_t h, w;
    float* data;
} dcl_matf;

typedef enum {NONE, FIRST, SECOND, BOTH} DCL_TRANSPOSE;

typedef struct dcl_matf_operations{
    void(*map)(const dcl_matf* A, float(*f)(float), dcl_matf* result);
    void(*add)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
    void(*sub)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result);
    void(*mul)(const dcl_matf* A, const dcl_matf* B, dcl_matf* result, DCL_TRANSPOSE option);
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
        ops->mul(layer->core, in, preout, NONE);
        ops->map(preout, layer->activation, out);
    } else 
        ops->map(in, layer->activation, out);
}

// void dcl_oerrorf(const dcl_matf* answer, const dcl_matf* out, dcl_matf* error, const dcl_matf_operations* ops){
//     ops->sub(answer, out, error);
// }
// void dcl_errorf(const dcl_matf* next_error, dcl_matf* preout, dcl_matf* error, const dcl_layerf* next_layer, const dcl_layerf* layer, const dcl_matf_operations* ops){
//     ops->mul(next_layer->core, next_error, error, FIRST);
//     ops->map(preout, layer->derivative, preout);
//     ops->had(error, preout, error);
// }
