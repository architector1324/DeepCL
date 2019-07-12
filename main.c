#include "deepcl.h"

// operations
void add(dcl_matf* A, dcl_matf* B, dcl_matf* result){
    size_t size = result->h * result->w;

    for(size_t i = 0; i < size; i++)
        result->data[i] = A->data[i] + B->data[i];
}

void sub(dcl_matf* A, dcl_matf* B, dcl_matf* result){
    size_t size = result->h * result->w;

    for(size_t i = 0; i < size; i++)
        result->data[i] = A->data[i] - B->data[i];
}

void had(dcl_matf* A, dcl_matf* B, dcl_matf* result){
    size_t size = result->h * result->w;

    for(size_t i = 0; i < size; i++)
        result->data[i] = A->data[i] * B->data[i];
}


dcl_matf_operations ops = {
    .map = NULL,
    .add = add,
    .sub = sub,
    .mul = NULL,
    .had = had
};


int main(){
    float _A[] = {1.0f, 2.0f, 3.0f, 4.0f};

    dcl_matf A = {
        .h = 2,
        .w = 2,
        .data = _A
    };

    return 0;
}