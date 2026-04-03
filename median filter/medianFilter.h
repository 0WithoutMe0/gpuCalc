#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include "utils.h"




class MedianFilter {
private:
    static float median_7(float arr[7]);
    static uint8_t median_9(uint8_t window[9]);
public:
    static void median_filter_7(const float* input, float* output, size_t length);
    static void median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride);
};



//-------------------------- 1D ћ≈ƒ»јЌЌџ… ‘»Ћ№“– ƒЋя ќ Ќј 1х7 --------------------------



//сортирующа€ сеть на 7 элементов (оптимальна€ верси€)
float MedianFilter::median_7(float arr[7]) {
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


//nosimd верси€ (в качестве сортировки все так же используем сортирующую сеть, так гораздо быстрее, чем стандартный std::sort)
void MedianFilter::median_filter_7(const float* input, float* output, size_t length) {
    float window[7];

    //первые 3 элемента
    for (size_t i = 0; i < 3; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = i + j;
            if (idx < 0) window[j + 3] = input[0];
            else if (idx >= length) window[j + 3] = input[length - 1];
            else window[j + 3] = input[idx];
        }

        output[i] = median_7(window);
    }

    //основна€ часть [3: length - 3]
    for (size_t i = 3; i < length - 3; ++i) {
        for (int j = -3; j <= 3; ++j) window[j + 3] = input[i + j];

        output[i] = median_7(window);
    }

    //последние 3 элемента
    for (size_t i = length - 3; i < length; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = i + j;
            if (idx < 0) window[j + 3] = input[0];
            else if (idx >= length) window[j + 3] = input[length - 1];
            else window[j + 3] = input[idx];
        }

        output[i] = median_7(window);
    }
}



//-------------------------- 2D ћ≈ƒ»јЌЌџ… ‘»Ћ№“– ƒЋя ќ Ќј 3х3 --------------------------



uint8_t MedianFilter::median_9(uint8_t window[9]) {
    cond_swap(window[0], window[3]);
    cond_swap(window[1], window[7]);
    cond_swap(window[2], window[5]);
    cond_swap(window[4], window[8]);

    cond_swap(window[0], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[8]);
    cond_swap(window[5], window[6]);

    window[2] = get_max(window[0], window[2]);
    cond_swap(window[1], window[3]);
    cond_swap(window[4], window[5]);
    window[7] = get_min(window[7], window[8]);

    window[4] = get_max(window[1], window[4]);
    window[3] = get_min(window[3], window[6]);
    window[5] = get_min(window[5], window[7]);

    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[5]);

    window[3] = get_max(window[2], window[3]);
    window[4] = get_min(window[4], window[5]);

    window[4] = get_max(window[3], window[4]);

    return window[4];
}


void MedianFilter::median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride) {
    //основной цикл
    for (size_t y = 0; y < height; ++y) {
        //3 р€да дл€ окна
        const uint8_t* y0 = input + (y > 0 ? y - 1 : 0) * stride;
        const uint8_t* y1 = input + y * stride;                   
        const uint8_t* y2 = input + (y < height - 1 ? y + 1 : y) * stride;

        //загружаем окно + считаем медиану
        for (size_t x = 0; x < width; ++x) {
            uint8_t window[9];

            size_t x0 = (x > 0 ? x - 1 : 0); 
            size_t x1 = x;                        
            size_t x2 = (x < width - 1 ? x + 1 : x); 

            window[0] = y0[x0]; window[1] = y0[x1]; window[2] = y0[x2];
            window[3] = y1[x0]; window[4] = y1[x1]; window[5] = y1[x2];
            window[6] = y2[x0]; window[7] = y2[x1]; window[8] = y2[x2];

            uint8_t median = median_9(window);

            output[y * stride + x] = median;
        }
    }
}