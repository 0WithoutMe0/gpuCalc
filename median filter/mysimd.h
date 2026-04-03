#pragma once

#include <algorithm>
#include <vector>
#include <cstring>
#include <immintrin.h>


#ifdef _MSC_VER
#define AVX2_FORCE_INLINE __forceinline
#else
#define AVX2_FORCE_INLINE __attribute__((always_inline)) inline
#endif



//-------------------------- INT 32 --------------------------

AVX2_FORCE_INLINE
__m256i load_vector_i32(const __m256i* arr) {
    return _mm256_loadu_si256(arr);
}

AVX2_FORCE_INLINE
__m256i fill_vector_i32(int v) {
    return _mm256_set1_epi32(v);
}

AVX2_FORCE_INLINE
__m256i set_vector_i32(int v7, int v6, int v5, int v4, int v3, int v2, int v1, int v0) {
    return _mm256_set_epi32(v7, v6, v5, v4, v3, v2, v1, v0);
}

AVX2_FORCE_INLINE
__m256i setr_vector_i32(int v7, int v6, int v5, int v4, int v3, int v2, int v1, int v0) {
    return _mm256_setr_epi32(v7, v6, v5, v4, v3, v2, v1, v0);
}

//приведение интового вектора к float без изменения бит внутри
AVX2_FORCE_INLINE
__m256 cast_i_to_f(__m256i mask) {
    return _mm256_castsi256_ps(mask);
}


//-------------------------- FLOAT --------------------------


AVX2_FORCE_INLINE 
__m256 fill_vector(float v) {
    return _mm256_set1_ps(v);
}

AVX2_FORCE_INLINE
__m256 set_vector(float v7, float v6, float v5, float v4, float v3, float v2, float v1, float v0) {
    return _mm256_set_ps(v7, v6, v5, v4, v3, v2, v1, v0);
}

AVX2_FORCE_INLINE 
__m256 load_vector(const float* arr) {
    return _mm256_load_ps(arr);
}

AVX2_FORCE_INLINE 
void store_vector(float* arr, __m256 r) {
    _mm256_store_ps(arr, r);
}

//переставить элементы вектора
AVX2_FORCE_INLINE
__m256 permute_vector(__m256 vec, __m256i permutation) {
    return _mm256_permutevar8x32_ps(vec, permutation);
}

//смешивание (маска известна на этапе копмиляции)
template<int MASK>
AVX2_FORCE_INLINE
__m256 blend_vector(__m256 a, __m256 b) {
    return _mm256_blend_ps(a, b, MASK);
}

//смешивание (маска является переменной)
AVX2_FORCE_INLINE
__m256 blendv_vector(__m256 a, __m256 b, __m256 mask) {
    return _mm256_blendv_ps(a, b, mask);
}

AVX2_FORCE_INLINE
void masked_store(float* arr, __m256 r, __m256i mask) {
    _mm256_maskstore_ps(arr, mask, r);
}

AVX2_FORCE_INLINE
__m256 masked_load(const float* arr, __m256i mask) {
    return _mm256_maskload_ps(arr, mask);
}

AVX2_FORCE_INLINE
__m256 masked_load_from(const float* arr, __m256 fill, __m256i mask) {
    __m256 values = masked_load(arr, mask);
    return blendv_vector(fill, values, cast_i_to_f(mask));
}

AVX2_FORCE_INLINE
__m256i make_loadmask(size_t value) {
    return _mm256_cmpgt_epi32(
        fill_vector_i32((int)value),
        setr_vector_i32(0, 1, 2, 3, 4, 5, 6, 7)
    );
}

AVX2_FORCE_INLINE
__m256 sum_vector(__m256 v1, __m256 v2) {
    return _mm256_add_ps(v1, v2);
}

AVX2_FORCE_INLINE 
__m256 min_vector(__m256 l, __m256 r) {
    return _mm256_min_ps(l, r);
}

AVX2_FORCE_INLINE 
__m256 max_vector(__m256 l, __m256 r) {
    return _mm256_max_ps(l, r);
}

AVX2_FORCE_INLINE  
void sort_pair(__m256& l, __m256& r) {
    __m256 tmp = min_vector(l, r);
    r = max_vector(l, r);
    l = tmp;
}

//вращение регистра вверх (циклический сдвиг)
template<int R>
AVX2_FORCE_INLINE 
__m256 rotate_up(__m256 r0) {
    static_assert(R >= 0 && R < 8);

    if constexpr (R == 0) return r0;
    else {
        //используем комбинацию перестановок
        constexpr int S = 8 - R;
        constexpr int A = (S + 0) % 8;
        constexpr int B = (S + 1) % 8;
        constexpr int C = (S + 2) % 8;
        constexpr int D = (S + 3) % 8;
        constexpr int E = (S + 4) % 8;
        constexpr int F = (S + 5) % 8;
        constexpr int G = (S + 6) % 8;
        constexpr int H = (S + 7) % 8;

        return permute_vector(r0, setr_vector_i32(A, B, C, D, E, F, G, H));
    }
}

//сдвиг вперед + перенос из другого регистра
template<int S>
AVX2_FORCE_INLINE 
__m256 shift_up_with_carry(__m256 prev, __m256 cur) {
    static_assert(S >= 0 && S <= 8);

    //маска для смещения: последние S элементов из prev, остальные из cur
    constexpr int mask = (0xFFu << (unsigned)S) & 0xFFu;
    return _mm256_blend_ps(rotate_up<S>(prev), rotate_up<S>(cur), mask);
}


//-------------------------- UNSIGNED INT 8 --------------------------


AVX2_FORCE_INLINE 
__m256i load_vector_8i(const uint8_t* arr) {
    return _mm256_loadu_si256((const __m256i*)arr);
}

AVX2_FORCE_INLINE 
void store_vector_8i(uint8_t* arr, __m256i r) {
    _mm256_storeu_si256((__m256i*)arr, r);
}

//загрузка со смещением (для соседних пикселей)
AVX2_FORCE_INLINE 
void load_8bit_neighbors(const uint8_t* src, __m256i a[3]) {
    a[0] = load_vector_8i(src - 1);  // левый столбец
    a[1] = load_vector_8i(src);      // центральный столбец
    a[2] = load_vector_8i(src + 1);  // правый столбец
}

AVX2_FORCE_INLINE 
__m256i min_vector_8u(__m256i a, __m256i b) {
    return _mm256_min_epu8(a, b);
}

AVX2_FORCE_INLINE 
__m256i max_vector_8u(__m256i a, __m256i b) {
    return _mm256_max_epu8(a, b);
}

AVX2_FORCE_INLINE 
void sort_pair_8u(__m256i& a, __m256i& b) {
    __m256i t = a;
    a = min_vector_8u(t, b);
    b = max_vector_8u(t, b);
}

//загрузка значений для окна 3х3
AVX2_FORCE_INLINE
void load_window_3x3_8u(const uint8_t* src, size_t stride, __m256i window[9]) {
    load_8bit_neighbors(src - stride, window + 0);  //верхние 3 элемента
    load_8bit_neighbors(src, window + 3);           //средние 3 элемента
    load_8bit_neighbors(src + stride, window + 6);  //нижние 3 элемента
}