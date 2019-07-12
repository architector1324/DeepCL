#include "deepcl.h"
#include "matrix_stub.h"


int main(){
    dcl_matf_operations ops = {
        .map = stub_map,
        .add = stub_add,
        .sub = stub_sub,
        .mul = stub_mul,
        .had = stub_had
    };

    float _A[] = {1.0f, 2.0f, 3.0f, 4.0f};

    dcl_matf A = {
        .h = 2,
        .w = 2,
        .data = _A
    };

    return 0;
}