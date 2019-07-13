#include "deepcl.h"
#include "matrix_stub.h"


float lrelu(float v){
    return v < 0 ? 0.01f * v : v;
}
float lrelu_div(float v){
    return v < 0 ? 0.01f : 1.0f;
}

int main(){
    dcl_matf_operations ops = {
        .map = stub_map,
        .add = stub_add,
        .sub = stub_sub,
        .mul = stub_mul,
        .had = stub_had
    };

    // setup data
    float _data[] = {-3.0f, 2.0f};
    dcl_matf data = {
        .h = 2, .w = 1,
        .data = _data
    };

    // setup containers
    float _il_out[2], _ol_preout[3], _ol_out[3];

    dcl_matf il_out = {
        .h = 2, .w = 1,
        .data = _il_out
    }; 

    dcl_matf ol_preout = {
        .h = 3, .w = 1,
        .data = _ol_preout
    };

    dcl_matf ol_out = {
        .h = 3, .w = 1,
        .data = _ol_out
    };

    float _ol_core[] = {1, 2, 3, 4, 5, 6};
    dcl_matf ol_core = {
        .h = 3, .w = 2,
        .data = _ol_core
    };

    // setup net
    dcl_layerf il = {
        .core = NULL,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    dcl_layerf ol = {
        .core = &ol_core,
        .activation = lrelu,
        .derivative = lrelu_div
    };

    // query
    dcl_queryf(&data, NULL, &il_out, &il, &ops);
    dcl_queryf(&il_out, &ol_preout, &ol_out, &ol, &ops);

    // output
    stub_printf(&data);
    stub_printf(&il_out);
    stub_printf(&ol_out);


    return 0;
}