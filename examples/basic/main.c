#include "deepcl.h"
#include "matrix_stub.h"


float lrelu(float v){
    return v < 0 ? 0.01f * v : v;
}
float lrelu_div(float v){
    return v < 0 ? 0.01f : 1.0f;
}

int main(){
    // setup data
    float _data[2] = {-3.0f, 2.0f};
    dcl_matf data = {
        .h = 2, .w = 1,
        .data = _data
    };

    float _answer[2] = {2.0f, -3.0f};
    dcl_matf answer = {
        .h = 2, .w = 1,
        .data = _answer
    };

    // setup containers
    float _il_out[2];
    float _hl_preout[3], _hl_out[3], _hl_error[3];
    float _ol_preout[2], _ol_out[2], _ol_error[2];

    dcl_matf il_out = {
        .h = 2, .w = 1,
        .data = _il_out
    }; 

    dcl_matf hl_preout = {
        .h = 3, .w = 1,
        .data = _hl_preout
    };

    dcl_matf hl_out = {
        .h = 3, .w = 1,
        .data = _hl_out
    };

    dcl_matf hl_error = {
        .h = 3, .w = 1,
        .data = _hl_error
    };

    float _hl_core[] = {1, 2, 3, 4, 5, 6};
    dcl_matf hl_core = {
        .h = 3, .w = 2,
        .data = _hl_core
    };

    dcl_matf ol_preout = {
        .h = 2, .w = 1,
        .data = _ol_preout
    };
    dcl_matf ol_out = {
        .h = 2, .w = 1,
        .data = _ol_out
    };
    dcl_matf ol_error = {
        .h = 2, .w = 1,
        .data = _ol_error
    };

    float _ol_core[] = {6, 5, 4, 3, 2, 1};
    dcl_matf ol_core = {
        .h = 2, .w = 3,
        .data = _ol_core
    };

    // setup net
    dcl_layerf il = {
        .core = NULL,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    dcl_layerf hl = {
        .core = &hl_core,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    dcl_layerf ol = {
        .core = &ol_core,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    // query
    dcl_queryf(&data, NULL, &il_out, &il, &stub_ops);
    dcl_queryf(&il_out, &hl_preout, &hl_out, &hl, &stub_ops);
    dcl_queryf(&hl_out, &ol_preout, &ol_out, &ol, &stub_ops);

    // error
    dcl_oerrorf(&answer, &ol_out, &ol_error, &stub_ops);
    dcl_errorf(&ol_error, &hl_preout, &hl_error, &ol, &hl, &stub_ops);

    // output
    puts("Data:");
    stub_printf(&data);

    puts("Output:");
    stub_printf(&il_out);
    stub_printf(&hl_out);
    stub_printf(&ol_out);

    puts("Error:");
    stub_printf(&hl_error);
    stub_printf(&ol_error);


    return 0;
}