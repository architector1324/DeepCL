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
typedef struct dio_matf{
    size_t h, w;
    float* data;
} dio_matf;

typedef enum {NONE, FIRST, SECOND, BOTH} DIO_TRANSPOSE;

typedef struct dio_matf_operations{
    void(*map)(const dio_matf* A, float(*f)(float), dio_matf* result, DIO_TRANSPOSE option);
    void(*add)(const dio_matf* A, const dio_matf* B, dio_matf* result, DIO_TRANSPOSE option);
    void(*sub)(const dio_matf* A, const dio_matf* B, dio_matf* result, DIO_TRANSPOSE option);
    void(*mul)(const dio_matf* A, const dio_matf* B, dio_matf* result, DIO_TRANSPOSE option);
    void(*had)(const dio_matf* A, const dio_matf* B, dio_matf* result, DIO_TRANSPOSE option);
    float(*map_reduce)(const dio_matf* A, float(*f)(float));
} dio_matf_operations;

#define DIO_WRAP_MAP(name) void (name)(const dio_matf* A, float(*f)(float), dio_matf* result, DIO_TRANSPOSE option)
#define DIO_WRAP_OPERATION(name) void (name)(const dio_matf* A, const dio_matf* B, dio_matf* result, DIO_TRANSPOSE option)
#define DIO_WRAP_MAP_REDUCE(name) float (name)(const dio_matf* A, float(*f)(float))

//////////////////////////////////
//             DEEP
//////////////////////////////////
// layer
typedef struct{
    dio_matf* core;

    float(*activation)(float);
    float(*derivative)(float);
} dio_layerf;



// API
void dio_queryf(const dio_matf* in, dio_matf* preout, dio_matf* out, const dio_layerf* layer, const dio_matf_operations* ops){
    if(layer->core && preout){
        ops->mul(layer->core, in, preout, NONE);
        ops->map(preout, layer->activation, out, NONE);
    } else 
        ops->map(in, layer->activation, out, NONE);
}

void dio_out_errorf(const dio_matf* answer, const dio_matf* out, dio_matf* error, const dio_matf_operations* ops){
    ops->sub(answer, out, error, NONE);
}
void dio_errorf(const dio_matf* next_error, dio_matf* preout, dio_matf* error, const dio_layerf* next_layer, const dio_layerf* layer, const dio_matf_operations* ops){
    ops->mul(next_layer->core, next_error, error, FIRST);
    ops->map(preout, layer->derivative, preout, NONE);
    ops->had(error, preout, error, NONE);
}

float dio_costf(const dio_matf* error, float(*cost)(float), const dio_matf_operations* ops){
    return ops->map_reduce(error, cost) / (error->h * error->w);
}