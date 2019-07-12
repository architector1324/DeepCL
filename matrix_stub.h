#pragma once

void stub_map(const dcl_matf* A, float(*f)(float), dcl_matf* result){
    size_t size = result->h * result->w;

    for(size_t i = 0; i < size; i++)
        result->data[i] = f(A->data[i]);
}

void stub_add(const dcl_matf* A, const dcl_matf* B, dcl_matf* result){
    size_t size = result->h * result->w;

    for(size_t i = 0; i < size; i++)
        result->data[i] = A->data[i] + B->data[i];
}

void stub_sub(const dcl_matf* A, const dcl_matf* B, dcl_matf* result){
    size_t size = result->h * result->w;

    for(size_t i = 0; i < size; i++)
        result->data[i] = A->data[i] - B->data[i];
}

void stub_mul(const dcl_matf* A, const dcl_matf* B, dcl_matf* result){

}

void stub_had(const dcl_matf* A, const dcl_matf* B, dcl_matf* result){
    size_t size = result->h * result->w;

    for(size_t i = 0; i < size; i++)
        result->data[i] = A->data[i] * B->data[i];
}