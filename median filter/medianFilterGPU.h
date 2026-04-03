#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include "utils.h"



class MedianFilterGPU {
private:
    static float median_7(float arr[7]);

public:
    static void median_filter_7(const float* input, float* output, size_t length);
};

//сортирующая сеть на 7 элементов
float MedianFilterGPU::median_7(float arr[7]) {
    cond_swap(arr[0], arr[6]);
    cond_swap(arr[2], arr[3]);
    cond_swap(arr[4], arr[5]);

    cond_swap(arr[0], arr[2]);
    cond_swap(arr[1], arr[4]);
    cond_swap(arr[3], arr[6]);

    arr[1] = get_max(arr[0], arr[1]);
    cond_swap(arr[2], arr[5]);
    cond_swap(arr[3], arr[4]);

    arr[2] = get_max(arr[1], arr[2]);
    arr[4] = get_min(arr[4], arr[6]);

    arr[3] = get_max(arr[2], arr[3]);
    arr[4] = get_min(arr[4], arr[5]);

    arr[3] = get_min(arr[3], arr[4]);

    return arr[3];
}

void MedianFilterGPU::median_filter_7(const float* input, float* output, size_t length) {
    sycl::queue q;

    size_t N = length;
    float* d_input = sycl::malloc_shared<float>(N, q);
    float* d_output = sycl::malloc_shared<float>(N, q);

    q.memcpy(d_input, input, N * sizeof(float)).wait();

    //центральные элементы вычисляем на стороне девайса
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N - 6), [=](sycl::id<1> idx) {
            size_t i = idx[0] + 3;//каждый поток обрабатывает свой индекс
            float window[7];//локальный массив для каждого потока

            for (int j = -3; j <= 3; ++j) window[j + 3] = d_input[i + j];

            d_output[i] = median_7(window);
        });
    });
    q.wait();

    //краевые элементы вычисляем на стороне хоста
    float window[7];

    //первые 3 элемента
    for (size_t i = 0; i < 3 && i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    //последние 3 элемента
    for (size_t i = (N > 3 ? N - 3 : 0); i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    //закончили вычисления
    q.memcpy(output, d_output, N * sizeof(float)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

