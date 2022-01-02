#include <time.h>
#include <math.h>

#include "deepio.h"
#include "matrix_stub.h"


// matrix operations wrapper
DIO_WRAP_INV_NEG_DIV(wrapped_inv_neg_div) {
    dio_data_t out;
    out.f = -1.0f / x;
    return out;
}

DIO_WRAP_DIV(wrapped_div) {
    dio_data_t out;
    out.f = x.f / y;
    return out;
}

DIO_WRAP_MAP(wrapped_stub_map) {
    stub_map(&A->data->f, f, &result->data->f, result->h, result->w);
}

DIO_WRAP_OPERATION(wrapped_stub_add) {
    stub_add(&A->data->f, &B->data->f, &result->data->f, result->h, result->w);
}

DIO_WRAP_OPERATION(wrapped_stub_sub) {
    stub_sub(&A->data->f, &B->data->f, &result->data->f, result->h, result->w);
}

DIO_WRAP_OPERATION(wrapped_stub_mul) {
    if(option == DIO_TRANSPOSE_NONE)
        stub_mul(&A->data->f, A->h, A->w, &B->data->f, B->h, B->w, &result->data->f, result->h, result->w, false, false);
    else if(option == DIO_TRANSPOSE_FIRST)
        stub_mul(&A->data->f, A->h, A->w, &B->data->f, B->h, B->w, &result->data->f, result->h, result->w, true, false);
    else if (option == DIO_TRANSPOSE_SECOND)
        stub_mul(&A->data->f, A->h, A->w, &B->data->f, B->h, B->w, &result->data->f, result->h, result->w, false, true);
}

DIO_WRAP_OPERATION(wrapped_stub_had) {
    stub_had(&A->data->f, &B->data->f, &result->data->f, result->h, result->w);
}

DIO_WRAP_MUL_SCALAR(wrapped_stub_mul_scalar) {
    stub_mul_scalar(&A->data->f, A->h, A->w, value.f, &result->data->f);
}

DIO_WRAP_MAP_REDUCE(wrapped_stub_map_reduce) {
    dio_data_t out;
    out.f = stub_map_reduce(&A->data->f, A->h, A->w, f);
    return out;
}

void wrapped_printf(const dio_mat_t* A) {
    stub_printf(&A->data->f, A->h, A->w);
}
//


float lrelu(float v) {
    return v < 0 ? 0.01f * v : v;
}
float lrelu_div(float v) {
    return v < 0 ? 0.01f : 1.0f;
}

float mse(float v) {
    return v * v;
}
float mse_div(float v) {
    return 2 * v;
}


int main() {
    srand(time(NULL));

    dio_mat_operations_t stub_ops = {
        .map = wrapped_stub_map,
        .add = wrapped_stub_add,
        .sub = wrapped_stub_sub,
        .mul = wrapped_stub_mul,
        .mul_scalar = wrapped_stub_mul_scalar,
        .had = wrapped_stub_had,
        .map_reduce = wrapped_stub_map_reduce,
        .inv_neg_div = wrapped_inv_neg_div,
        .div = wrapped_div
    };

    // setup data
    dio_mat_t data = {
        .h = 2, .w = 1,
        .data = (float[2]) {-3.0f, 2.0f}
    };

    dio_mat_t answer = {
        .h = 2, .w = 1,
        .data = (float[2]) {2.0f, -3.0f}
    };

    // setup net
    dio_layer_t il = {
        .core = NULL,
        .neuronsCount = 2,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    dio_mat_t hl_core;

    dio_layer_t hl = {
        .core = &hl_core,
        .neuronsCount = 3,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    dio_layer_init(&il, &hl);
    hl_core.data = (dio_data_t*)malloc(hl_core.w * hl_core.h * sizeof(float));
    stub_rnd(0.0f, 1.0f, &hl_core.data->f, hl_core.h, hl_core.w);

    dio_mat_t ol_core;

    dio_layer_t ol = {
        .core = &ol_core,
        .neuronsCount = 2,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    dio_layer_init(&hl, &ol);
    ol_core.data = (dio_data_t*)malloc(ol_core.w * ol_core.h * sizeof(float));
    stub_rnd(0.0f, 1.0f, &ol_core.data->f, ol_core.h, ol_core.w);

    // setup containers
    dio_stock_t il_stock;
    dio_stock_t hl_stock;
    dio_stock_t ol_stock;

    dio_stock_init(&il, 1, &il_stock);
    dio_stock_init(&hl, 1, &hl_stock);
    dio_stock_init(&ol, 1, &ol_stock);

    il_stock.preout.data = (dio_data_t*)malloc(il_stock.preout.w * il_stock.preout.h * sizeof(float));
    il_stock.out.data = (dio_data_t*)malloc(il_stock.out.w * il_stock.out.h * sizeof(float));
    il_stock.error.data = (dio_data_t*)malloc(il_stock.error.w * il_stock.error.h * sizeof(float));
    il_stock.grad.data = (dio_data_t*)malloc(il_stock.grad.w * il_stock.grad.h * sizeof(float));

    hl_stock.preout.data = (dio_data_t*)malloc(hl_stock.preout.w * hl_stock.preout.h * sizeof(float));
    hl_stock.out.data = (dio_data_t*)malloc(hl_stock.out.w * hl_stock.out.h * sizeof(float));
    hl_stock.error.data = (dio_data_t*)malloc(hl_stock.error.w * hl_stock.error.h * sizeof(float));
    hl_stock.grad.data = (dio_data_t*)malloc(hl_stock.grad.w * hl_stock.grad.h * sizeof(float));

    ol_stock.preout.data = (dio_data_t*)malloc(ol_stock.preout.w * ol_stock.preout.h * sizeof(float));
    ol_stock.out.data = (dio_data_t*)malloc(ol_stock.out.w * ol_stock.out.h * sizeof(float));
    ol_stock.error.data = (dio_data_t*)malloc(ol_stock.error.w * ol_stock.error.h * sizeof(float));
    ol_stock.grad.data = (dio_data_t*)malloc(ol_stock.grad.w * ol_stock.grad.h * sizeof(float));

    // fit
    float error = 1.0f;
    float eLim = 0.0005f;

    while(error > eLim) {
        // query
        dio_stock_query(&data, &il_stock, &stub_ops);
        dio_stock_query(&il_stock.out, &hl_stock, &stub_ops);
        dio_stock_query(&hl_stock.out, &ol_stock, &stub_ops);

        // error
        dio_stock_out_error(&answer, &ol_stock, &stub_ops);
        dio_stock_error(&ol_stock, &hl_stock, &stub_ops);

        error = dio_stock_cost(&ol_stock, mse, &stub_ops).f;

        // grad
        dio_stock_grad(&hl_stock, &il_stock, mse_div, &stub_ops);
        dio_stock_grad(&ol_stock, &hl_stock, mse_div, &stub_ops);

        // train
        dio_stock_basic_gd(&hl_stock, (dio_data_t)0.1f, &stub_ops);
        dio_stock_basic_gd(&ol_stock, (dio_data_t)0.1f, &stub_ops);

        printf("Total error: %f\n", error);
    }

    // output
    puts("Data:");
    wrapped_printf(&data);

    puts("Output:");
    wrapped_printf(&il_stock.out);
    wrapped_printf(&hl_stock.out);
    wrapped_printf(&ol_stock.out);
    
    puts("Answer:");
    wrapped_printf(&answer);

    // free
    free(hl.core->data);
    free(ol.core->data);

    free(il_stock.preout.data);
    free(il_stock.out.data);
    free(il_stock.error.data);
    free(il_stock.grad.data);

    free(hl_stock.preout.data);
    free(hl_stock.out.data);
    free(hl_stock.error.data);
    free(hl_stock.grad.data);

    free(ol_stock.preout.data);
    free(ol_stock.out.data);
    free(ol_stock.error.data);
    free(ol_stock.grad.data);

    return 0;
}
