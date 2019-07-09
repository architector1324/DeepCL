#include "deepcl.h"

int main(){
    float _A[] = {1.0f, 2.0f, 3.0f, 4.0f};

    dcl_matf A = {
        .h = 2,
        .w = 2,
        .data = _A
    };

    return 0;
}