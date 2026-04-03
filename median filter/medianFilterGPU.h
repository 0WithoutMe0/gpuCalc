#pragma once

#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include "utils.h"

class MedianFilterGPU {
private:
    static float median_7(float arr[7]);
    static uint8_t median_9(uint8_t arr[9]);

public:
    static void median_filter_7(const float* input, float* output, size_t length);
    static void median_filter_3x3(const uint8_t* input, uint8_t* output,
                                  size_t width, size_t height, size_t stride);
};

// -------------------- 1D --------------------

inline float MedianFilterGPU::median_7(float arr[7]) {
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

inline void MedianFilterGPU::median_filter_7(const float* input, float* output, size_t length) {
    sycl::queue q{sycl::default_selector_v};

    size_t N = length;
    float* d_input = sycl::malloc_shared<float>(N, q);
    float* d_output = sycl::malloc_shared<float>(N, q);

    q.memcpy(d_input, input, N * sizeof(float)).wait();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N - 6), [=](sycl::id<1> idx) {
            size_t i = idx[0] + 3;
            float window[7];

            for (int j = -3; j <= 3; ++j) window[j + 3] = d_input[i + j];

            d_output[i] = median_7(window);
        });
    }).wait();

    float window[7];

    for (size_t i = 0; i < 3 && i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    for (size_t i = (N > 3 ? N - 3 : 0); i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    q.memcpy(output, d_output, N * sizeof(float)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

// -------------------- 2D --------------------

inline uint8_t MedianFilterGPU::median_9(uint8_t window[9]) {
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

inline void MedianFilterGPU::median_filter_3x3(const uint8_t* input, uint8_t* output,
                                                size_t width, size_t height, size_t stride) {
    if (width == 0 || height == 0) return;

    constexpr size_t BLOCK_Y = 16;
    constexpr size_t BLOCK_X = 16;

    sycl::queue q{sycl::default_selector_v};

    const size_t element_count = height * stride;
    const size_t total_bytes = element_count * sizeof(uint8_t);

    uint8_t* d_input = sycl::malloc_device<uint8_t>(element_count, q);
    uint8_t* d_output = sycl::malloc_device<uint8_t>(element_count, q);

    q.memcpy(d_input, input, total_bytes).wait();
    q.memset(d_output, 0, element_count).wait();

    const size_t global_y = ((height + BLOCK_Y - 1) / BLOCK_Y) * BLOCK_Y;
    const size_t global_x = ((width + BLOCK_X - 1) / BLOCK_X) * BLOCK_X;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> tile(sycl::range<2>(BLOCK_Y + 2, BLOCK_X + 2), h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(global_y, global_x),
                sycl::range<2>(BLOCK_Y, BLOCK_X)
            ),
            [=](sycl::nd_item<2> item) {
                const size_t gy = item.get_global_id(0);
                const size_t gx = item.get_global_id(1);

                const size_t ly = item.get_local_id(0);
                const size_t lx = item.get_local_id(1);

                const size_t base_y = item.get_group(0) * BLOCK_Y;
                const size_t base_x = item.get_group(1) * BLOCK_X;

                // Кооперативная загрузка тайла с halo = 1
                for (size_t ty = ly; ty < BLOCK_Y + 2; ty += BLOCK_Y) {
                    int src_y = static_cast<int>(base_y + ty) - 1;
                    if (src_y < 0) src_y = 0;
                    if (src_y >= static_cast<int>(height)) src_y = static_cast<int>(height) - 1;

                    for (size_t tx = lx; tx < BLOCK_X + 2; tx += BLOCK_X) {
                        int src_x = static_cast<int>(base_x + tx) - 1;
                        if (src_x < 0) src_x = 0;
                        if (src_x >= static_cast<int>(width)) src_x = static_cast<int>(width) - 1;

                        tile[ty][tx] = d_input[src_y * stride + src_x];
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

                if (gy < height && gx < width) {
                    uint8_t window[9];
                    int idx = 0;

                    for (int dy = 0; dy < 3; ++dy) {
                        for (int dx = 0; dx < 3; ++dx) {
                            window[idx++] = tile[ly + dy][lx + dx];
                        }
                    }

                    d_output[gy * stride + gx] = median_9(window);
                }
            }
        );
    }).wait();

    q.memcpy(output, d_output, total_bytes).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}