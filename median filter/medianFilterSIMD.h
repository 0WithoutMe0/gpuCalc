#pragma once

#include <cstdint>
#include <immintrin.h>
#include "mysimd.h"

class MedianFilterSIMD {
private:
    static __m256 get_vector_of_median_7(__m256 s1, __m256 hi);
    static void get_vector_of_median_3x3_8u(__m256i window[9]);
public:
    static void median_filter_7(const float* input, float* output, size_t length);
    static void median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride);
};



//-------------------------- 1D МЕДИАННЫЙ ФИЛЬТР ДЛЯ ОКНА 1х7 --------------------------



//медиана для окна 7 (находит сразу 8 медианных элементов)
//принцип как в обычной версии, только вместо 1 элемента - сразу вектор,
//а вместо одного сравнения cond_swap мы применяем аналогичную операцию, но сразу к 8 эл-нтам
AVX2_FORCE_INLINE
__m256 MedianFilterSIMD::get_vector_of_median_7(__m256 s1, __m256 hi) {
    //окно из 7 элементов, только в качестве элементов выступают вектора
    __m256 s2 = shift_up_with_carry<7>(s1, hi);
    __m256 s3 = shift_up_with_carry<6>(s1, hi);
    __m256 s4 = shift_up_with_carry<5>(s1, hi);
    __m256 s5 = shift_up_with_carry<4>(s1, hi);
    __m256 s6 = shift_up_with_carry<3>(s1, hi);
    __m256 s7 = shift_up_with_carry<2>(s1, hi);

    //сортирующая сеть для векторов (на выходе вектор из 8 медиан)
    sort_pair(s1, s7);
    sort_pair(s3, s4);
    sort_pair(s5, s6);

    sort_pair(s1, s3);
    sort_pair(s2, s5);
    sort_pair(s4, s7);

    s2 = max_vector(s1, s2);
    sort_pair(s3, s6);
    sort_pair(s4, s5);

    s3 = max_vector(s2, s3);
    s5 = min_vector(s5, s7);

    s4 = max_vector(s3, s4);
    s5 = min_vector(s5, s6);

    s4 = min_vector(s4, s5);

    return s4;
}

//медианный фильтр размера 7
void MedianFilterSIMD::median_filter_7(const float* input, float* output, size_t length) {
    __m256          prev;   //нижняя часть окна
    __m256          curr;   //средняя часть окна
    __m256          next;   //верхняя часть окна
    __m256          lo;     //нижний регистр для расчета медианы (покрывает последовательность слева)
    __m256          hi;     //верхний регистр для расчета медианы (покрывает последовательность справа)
    __m256i         mask;   //маска для хвостовых элементов (если осталось меньше 8 значений)
    __m256          median7;//результат вычисления медианы 7 элементов

    //граничные значения (для выхода за границы массива)
    __m256 first = fill_vector(input[0]);
    __m256 last = fill_vector(input[length - 1]);

    //обработка массива
    size_t read = 0;
    size_t wrote = 0;

    prev = first;
    curr = load_vector(input);
    read += 8;

    //основной цикл (пока не обработали все значения)
    while (wrote < length) {
        //считываем данные из input
        //обработка основной части (если можно загрузить 8 элементов)
        if (read <= (length - 8)) {
            next = load_vector(input + read);
            read += 8;
        }
        //обработка хвоста (осталось меньше 8 необработанных элементов)
        else {
            mask = make_loadmask(length - read);
            next = masked_load_from(input + read, last, mask);
            read = length;
        }

        //формируем окно и вычисляем медиану
        lo = shift_up_with_carry<3>(prev, curr);
        hi = shift_up_with_carry<3>(curr, next);
        median7 = get_vector_of_median_7(lo, hi);

        //записываем результат вычислений в output
        //обработка основной части (если можно загрузить 8 элементов)
        if (wrote <= (length - 8)) {
            store_vector(output + wrote, median7);
            wrote += 8;
        }
        //обработка хвоста (осталось меньше 8 необработанных элементов)
        else {
            mask = make_loadmask(length - wrote);
            masked_store(output + wrote, median7, mask);
            wrote = length;
        }

        //сдвигаем окно
        prev = curr;
        curr = next;
    }
}



//-------------------------- 2D МЕДИАННЫЙ ФИЛЬТР ДЛЯ ОКНА 3х3 --------------------------



//поиск медианы для окна 3х3
AVX2_FORCE_INLINE
void MedianFilterSIMD::get_vector_of_median_3x3_8u(__m256i a[9]) {
    sort_pair_8u(a[0], a[3]);
    sort_pair_8u(a[1], a[7]);
    sort_pair_8u(a[2], a[5]);
    sort_pair_8u(a[4], a[8]);

    sort_pair_8u(a[0], a[7]);
    sort_pair_8u(a[2], a[4]);
    sort_pair_8u(a[3], a[8]);
    sort_pair_8u(a[5], a[6]);

    a[2] = max_vector_8u(a[0], a[2]);
    sort_pair_8u(a[1], a[3]);
    sort_pair_8u(a[4], a[5]);
    a[7] = min_vector_8u(a[7], a[8]);

    a[4] = max_vector_8u(a[1], a[4]);
    a[3] = min_vector_8u(a[3], a[6]);
    a[5] = min_vector_8u(a[5], a[7]);

    sort_pair_8u(a[2], a[4]);
    sort_pair_8u(a[3], a[5]);

    a[3] = max_vector_8u(a[2], a[3]);
    a[4] = min_vector_8u(a[4], a[5]);

    a[4] = max_vector_8u(a[3], a[4]);
}


void MedianFilterSIMD::median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride) {
    //расширенное изображение (+1 строка и +1 столбец по краям, чтобы корректно работал фильтр)
    const size_t padded_width = width + 2;
    const size_t padded_height = height + 2;
    std::vector<uint8_t> padded(padded_width * padded_height);

    //центральная часть
    for (size_t y = 0; y < height; ++y) {
        const uint8_t* src_row = input + y * stride;
        uint8_t* dst_row = padded.data() + (y + 1) * padded_width + 1;
        memcpy(dst_row, src_row, width);
    }

    //заполняем границы методом зеркального отражения (дублируем то же, что и по краям)
    //верхняя и нижняя границы
    for (size_t x = 0; x < width; ++x) {
        padded[(0) * padded_width + (x + 1)] = input[(0) * stride + x];
        padded[(padded_height - 1) * padded_width + (x + 1)] = input[(height - 1) * stride + x];
    }
    //левая и правая границы
    for (size_t y = 0; y < height; ++y) {
        padded[(y + 1) * padded_width + 0] = input[y * stride + 0];
        padded[(y + 1) * padded_width + padded_width - 1] = input[y * stride + width - 1];
    }

    //углы
    padded[0] = input[0];
    padded[padded_width - 1] = input[width - 1];
    padded[(padded_height - 1) * padded_width] = input[(height - 1) * stride];
    padded[(padded_height - 1) * padded_width + padded_width - 1] = input[(height - 1) * stride + width - 1];

    //основной цикл
    __m256i window[9];

    for (size_t y = 0; y < height; ++y) {
        const uint8_t* src_row = padded.data() + (y + 1) * padded_width + 1;
        uint8_t* dst_row = output + y * stride;

        size_t x = 0;
        //обрабатываем полные блоки по 32 пикселя (32 * sizeof(uint8) = 256)
        for (; x + 32 <= width; x += 32) {
            load_window_3x3_8u(src_row + x, padded_width, window);
            get_vector_of_median_3x3_8u(window);
            store_vector_8i(dst_row + x, window[4]);
        }

        //хвост (последние <32 пикселей)
        if (x < width) {
            for (; x < width; ++x) {
                uint8_t pixels[9];
                size_t idx = 0;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        size_t src_x = x + dx;
                        size_t src_y = y + dy + 1;
                        pixels[idx++] = padded[src_y * padded_width + (src_x + 1)];
                    }
                }

                std::sort(pixels, pixels + 9);
                dst_row[x] = pixels[4];
            }
        }
    }
}