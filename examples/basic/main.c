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
    stub_map_reduce(&A->data->f, A->h, A->w, f);
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

    // setup containers
    dio_mat_t il_out = {
        .h = 2, .w = 1,
        .data = (float[2]) {}
    }; 

    dio_mat_t hl_preout = {
        .h = 3, .w = 1,
        .data = (float[3]) {}
    };
    dio_mat_t hl_out = {
        .h = 3, .w = 1,
        .data = (float[3]) {}
    };
    dio_mat_t hl_error = {
        .h = 3, .w = 1,
        .data = (float[3]) {}
    };
    dio_mat_t hl_grad = {
        .h = 3, .w = 2,
        .data = (float[6]) {}
    };
    dio_mat_t hl_core = {
        .h = 3, .w = 2,
        .data = (float[6]) {1, 2, 3, 4, 5, 6}
    };

    dio_mat_t ol_preout = {
        .h = 2, .w = 1,
        .data = (float[2]) {}
    };
    dio_mat_t ol_out = {
        .h = 2, .w = 1,
        .data = (float[2]) {}
    };
    dio_mat_t ol_error = {
        .h = 2, .w = 1,
        .data = (float[2]) {}
    };
    dio_mat_t ol_grad = {
        .h = 2, .w = 3,
        .data = (float[6]) {}
    };
    dio_mat_t ol_core = {
        .h = 2, .w = 3,
        .data = (float[6]) {6, 5, 4, 3, 2, 1}
    };

    // setup net
    dio_layer_t il = {
        .core = NULL,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    dio_layer_t hl = {
        .core = &hl_core,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    dio_layer_t ol = {
        .core = &ol_core,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    for(size_t i = 0; i < 200; i++) {
        // query
        dio_query(&data, NULL, &il_out, &il, &stub_ops);
        dio_query(&il_out, &hl_preout, &hl_out, &hl, &stub_ops);
        dio_query(&hl_out, &ol_preout, &ol_out, &ol, &stub_ops);

        // error
        dio_out_error(&answer, &ol_out, &ol_error, &stub_ops);
        dio_error(&ol_error, &hl_preout, &hl_error, &ol, &hl, &stub_ops);

        float error = dio_cost(&ol_error, mse, &stub_ops).f;

        // grad
        dio_grad(&hl_error, &il_out, &hl_grad, mse_div, &stub_ops);
        dio_grad(&ol_error, &hl_out, &ol_grad, mse_div, &stub_ops);

        // train
        dio_basic_gd(&hl_grad, &hl, (dio_data_t)0.1f, &stub_ops);
        dio_basic_gd(&ol_grad, &ol, (dio_data_t)0.1f, &stub_ops);

        if(i % 10 == 0)
            printf("Total error: %f\n", error);
    }

    // output
    puts("Data:");
    wrapped_printf(&data);

    puts("Output:");
    // wrapped_printf(&il_out);
    // wrapped_printf(&hl_out);
    wrapped_printf(&ol_out);
    
    puts("Answer:");
    wrapped_printf(&answer);

    return 0;
}