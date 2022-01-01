#pragma once
#include <stddef.h>
#include <stdlib.h>


/* 
    You can use this library on any device (include embedded systems), no memory allocations at all!
    Also you can use any non-integer numbers representation (float, fixed, etc...)
    
*/ 

//////////////////////////////////
//            MATRIX
//////////////////////////////////
typedef union {
    float f;
    double d;
    int32_t fxd32;
    int16_t fxd16;
} dio_data_t;


typedef struct dio_mat_t{
    size_t h, w;
    dio_data_t* data;
} dio_mat_t;

typedef enum {NONE, FIRST, SECOND, BOTH} DIO_TRANSPOSE;

typedef struct dio_mat_operations_t{
    void(*map)(const dio_mat_t* A, dio_data_t(*f)(dio_data_t), dio_mat_t* result, DIO_TRANSPOSE option);
    void(*add)(const dio_mat_t* A, const dio_mat_t* B, dio_mat_t* result, DIO_TRANSPOSE option);
    void(*sub)(const dio_mat_t* A, const dio_mat_t* B, dio_mat_t* result, DIO_TRANSPOSE option);
    void(*mul)(const dio_mat_t* A, const dio_mat_t* B, dio_mat_t* result, DIO_TRANSPOSE option);
    void(*mul_scalar)(const dio_mat_t* A, dio_data_t value, dio_mat_t* result, DIO_TRANSPOSE option);
    void(*had)(const dio_mat_t* A, const dio_mat_t* B, dio_mat_t* result, DIO_TRANSPOSE option);
    dio_data_t(*map_reduce)(const dio_mat_t* A, dio_data_t(*f)(dio_data_t));
    dio_data_t(*inv_neg_div)(size_t x);
    dio_data_t(*div)(dio_data_t x, size_t y);
} dio_mat_operations_t;


// -1/x
#define DIO_WRAP_INV_NEG_DIV(name) dio_data_t name(size_t x)
#define DIO_WRAP_DIV(name) dio_data_t name(dio_data_t x, size_t y)

#define DIO_WRAP_OPERATION(name) void name(const dio_mat_t* A, const dio_mat_t* B, dio_mat_t* result, DIO_TRANSPOSE option)
#define DIO_WRAP_MAP(name) void name(const dio_mat_t* A, dio_data_t(*f)(dio_data_t), dio_mat_t* result, DIO_TRANSPOSE option)
#define DIO_WRAP_MUL_SCALAR(name) void name(const dio_mat_t* A, dio_data_t value, dio_mat_t* result, DIO_TRANSPOSE option)
#define DIO_WRAP_MAP_REDUCE(name) dio_data_t name(const dio_mat_t* A, dio_data_t(*f)(dio_data_t))

//////////////////////////////////
//             DEEP
//////////////////////////////////
// layer
typedef struct{
    dio_mat_t* core;

    dio_data_t(*activation)(dio_data_t);
    dio_data_t(*derivative)(dio_data_t);
} dio_layerf;



// API
void dio_query(const dio_mat_t* in, dio_mat_t* preout, dio_mat_t* out, const dio_layerf* layer, const dio_mat_operations_t* ops){
    if(layer->core && preout){
        ops->mul(layer->core, in, preout, NONE);
        ops->map(preout, layer->activation, out, NONE);
    } else 
        ops->map(in, layer->activation, out, NONE);
}

void dio_out_error(const dio_mat_t* answer, const dio_mat_t* out, dio_mat_t* error, const dio_mat_operations_t* ops){
    ops->sub(answer, out, error, NONE);
}

void dio_error(const dio_mat_t* next_error, dio_mat_t* preout, dio_mat_t* error, const dio_layerf* next_layer, const dio_layerf* layer, const dio_mat_operations_t* ops){
    ops->mul(next_layer->core, next_error, error, FIRST);
    ops->map(preout, layer->derivative, preout, NONE);
    ops->had(error, preout, error, NONE);
}

dio_data_t dio_cost(const dio_mat_t* error, dio_data_t(*cost)(dio_data_t), const dio_mat_operations_t* ops){
    return ops->div(ops->map_reduce(error, cost), error->h * error->w);
}

void dio_grad(dio_mat_t* error, const dio_mat_t* prev_out, dio_mat_t* grad, dio_data_t(*div_cost)(dio_data_t), const dio_mat_operations_t* ops){
    dio_data_t count = ops->inv_neg_div(error->h * error->w);

    ops->map(error, div_cost, error, NONE);
    ops->mul(error, prev_out, grad, SECOND);
    ops->mul_scalar(grad, count, grad, NONE);
}

// basic gradient discent
void dio_basic_gd(dio_mat_t* grad, const dio_layerf* layer, dio_data_t learning_rate, const dio_mat_operations_t* ops){
    ops->mul_scalar(grad, learning_rate, grad, NONE);
    ops->sub(layer->core, grad, layer->core, NONE);
}