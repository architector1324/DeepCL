#pragma once
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>


// Super-duper matrix library (just a stub)

void stub_map(const float* A, float(*f)(float), float* result, size_t result_h, size_t result_w){
    size_t size = result_h * result_w;

    for(size_t i = 0; i < size; i++)
        result[i] = f(A[i]);
}

void stub_add(const float* A, const float* B, float* result, size_t result_h, size_t result_w){
    size_t size = result_h * result_w;

    for(size_t i = 0; i < size; i++)
        result[i] = A[i] + B[i];
}

void stub_sub(const float* A, const float* B, float* result, size_t result_h, size_t result_w){
    size_t size = result_h * result_w;

    for(size_t i = 0; i < size; i++)
        result[i] = A[i] - B[i];
}

void stub_mul(const float* A, size_t A_h, size_t A_w,const float* B, size_t B_h, size_t B_w,float* result, size_t result_h, size_t result_w, bool transpose_first, bool transpose_second){
    if(!transpose_first && !transpose_second){
        size_t h = A_h;
        size_t w = B_w;

        for(size_t i = 0; i < h; i++){
            size_t A_offset = i * A_w;

            for(size_t j = 0; j < w; j++){
                float sum = 0;
                for(size_t k = 0; k < A_w; k++)
                    sum += A[A_offset + k] * B[k * w + j];

                result[i * w + j] = sum;
            }
        }
    }else if(transpose_first){
        size_t h = A_w;
        size_t w = B_w;
        size_t _A_w = A_h;

        for(size_t i = 0; i < h; i++){
            for(size_t j = 0; j < w; j++){
                float sum = 0;
                for(size_t k = 0; k < _A_w; k++)
                    sum += A[k * h + i] * B[k * w + j];

                result[i * w + j] = sum;
            }
        }
    }else if(transpose_second){
        size_t h = A_h;
        size_t w = B_h;

        for(size_t i = 0; i < h; i++){
            size_t A_offset = i * A_w;

            for(size_t j = 0; j < w; j++){
                float sum = 0;
                for(size_t k = 0; k < A_w; k++)
                    sum += A[A_offset + k] * B[j * B_w + k];

                result[i * w + j] = sum;
            }
        }
    }
}

void stub_had(const float* A, const float* B, float* result, size_t result_h, size_t result_w){
    size_t size = result_h * result_w;

    for(size_t i = 0; i < size; i++) result[i] = A[i] * B[i];
}

float stub_map_reduce(const float* A, size_t A_h, size_t A_w, float(*f)(float)){
    float result = 0.0f;
    size_t size = A_h * A_w;

    for(size_t i = 0; i < size; i++) result += f(A[i]);

    return result;
}

void stub_mul_scalar(const float* A, size_t A_h, size_t A_w, float value, float* result){
    size_t size = A_h * A_w;
    for(size_t i = 0; i < size; i++) result[i] = value * A[i];
}

void stub_printf(const float* A, size_t A_h, size_t A_w){
    size_t h = A_h;
    size_t w = A_w;

    for(size_t i = 0; i < h; i++){
        size_t offset = i * w;
        for(size_t j = 0; j < w; j++)
            printf("%f ", A[offset + j]);
        puts("");
    }
    puts("");
}

void stub_rnd(float from, float to, float* result, size_t result_h, size_t result_w) {
    size_t size = result_h * result_w;
    for(size_t i = 0; i < size; i++) result[i] = from + ((float)rand() / RAND_MAX) * (to - from);
}
