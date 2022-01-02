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

typedef enum {DIO_TRANSPOSE_NONE, DIO_TRANSPOSE_FIRST, DIO_TRANSPOSE_SECOND, DIO_TRANSPOSE_BOTH} DIO_TRANSPOSE;

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
    size_t neuronsCount;

    dio_data_t(*activation)(dio_data_t);
    dio_data_t(*derivative)(dio_data_t);
} dio_layer_t;

// stock
typedef struct {
    dio_mat_t preout;
    dio_mat_t out;
    dio_mat_t error;

    dio_mat_t grad;

    const dio_layer_t* layer;
} dio_stock_t;


// API
void dio_layer_init(const dio_layer_t* prevLayer, dio_layer_t* layer) {
    layer->core->w = prevLayer->neuronsCount;
    layer->core->h = layer->neuronsCount;
}

void dio_stock_init(const dio_layer_t* layer, size_t examplesCount, dio_stock_t* stock) {
    stock->preout.w = examplesCount;
    stock->preout.h = layer->neuronsCount;

    stock->out.w = examplesCount;
    stock->out.h = layer->neuronsCount;

    stock->error.w = examplesCount;
    stock->error.h = layer->neuronsCount;

    if(layer->core) {
        stock->grad.w = layer->core->w;
        stock->grad.h = layer->core->h;
    }

    stock->layer = layer;
}


void dio_query(const dio_mat_t* in, dio_mat_t* preout, dio_mat_t* out, const dio_layer_t* layer, const dio_mat_operations_t* ops){
    if(layer->core && preout){
        ops->mul(layer->core, in, preout, DIO_TRANSPOSE_NONE);
        ops->map(preout, layer->activation, out, DIO_TRANSPOSE_NONE);
    } else 
        ops->map(in, layer->activation, out, DIO_TRANSPOSE_NONE);
}

void dio_stock_query(const dio_mat_t* in, dio_stock_t* stock, const dio_mat_operations_t* ops) {
    dio_query(in, &stock->preout, &stock->out, stock->layer, ops);
}

void dio_out_error(const dio_mat_t* answer, const dio_mat_t* out, dio_mat_t* error, const dio_mat_operations_t* ops){
    ops->sub(answer, out, error, DIO_TRANSPOSE_NONE);
}

void dio_stock_out_error(const dio_mat_t* answer, dio_stock_t* stock, const dio_mat_operations_t* ops) {
    dio_out_error(answer, &stock->out, &stock->error, ops);
}

void dio_error(const dio_mat_t* next_error, dio_mat_t* preout, dio_mat_t* error, const dio_layer_t* next_layer, const dio_layer_t* layer, const dio_mat_operations_t* ops){
    ops->mul(next_layer->core, next_error, error, DIO_TRANSPOSE_FIRST);
    ops->map(preout, layer->derivative, preout, DIO_TRANSPOSE_NONE);
    ops->had(error, preout, error, DIO_TRANSPOSE_NONE);
}

void dio_stock_error(const dio_stock_t* nextStock, dio_stock_t* stock, const dio_mat_operations_t* ops) {
    dio_error(&nextStock->error, &stock->preout, &stock->error, nextStock->layer, stock->layer, ops);
}

dio_data_t dio_cost(const dio_mat_t* error, dio_data_t(*cost)(dio_data_t), const dio_mat_operations_t* ops){
    return ops->div(ops->map_reduce(error, cost), error->h * error->w);
}

dio_data_t dio_stock_cost(const dio_stock_t* stock, dio_data_t(*cost)(dio_data_t), const dio_mat_operations_t* ops) {
    dio_cost(&stock->error, cost, ops);
}

void dio_grad(dio_mat_t* error, const dio_mat_t* prev_out, dio_mat_t* grad, dio_data_t(*div_cost)(dio_data_t), const dio_mat_operations_t* ops){
    dio_data_t count = ops->inv_neg_div(error->h * error->w);

    ops->map(error, div_cost, error, DIO_TRANSPOSE_NONE);
    ops->mul(error, prev_out, grad, DIO_TRANSPOSE_SECOND);
    ops->mul_scalar(grad, count, grad, DIO_TRANSPOSE_NONE);
}

void dio_stock_grad(dio_stock_t* stock, const dio_stock_t* prevStock, dio_data_t(*div_cost)(dio_data_t), const dio_mat_operations_t* ops) {
    dio_grad(&stock->error, &prevStock->out, &stock->grad, div_cost, ops);
}

// basic gradient discent
void dio_basic_gd(dio_mat_t* grad, const dio_layer_t* layer, dio_data_t learning_rate, const dio_mat_operations_t* ops){
    ops->mul_scalar(grad, learning_rate, grad, DIO_TRANSPOSE_NONE);
    ops->sub(layer->core, grad, layer->core, DIO_TRANSPOSE_NONE);
}

void dio_stock_basic_gd(dio_stock_t* stock, dio_data_t learning_rate, const dio_mat_operations_t* ops) {
    dio_basic_gd(&stock->grad, stock->layer, learning_rate, ops);
}