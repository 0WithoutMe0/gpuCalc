#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <cassert>

#include "medianFilter.h"
#include "medianFilterSIMD.h"
#include "medianFilterGPU.h"

namespace {

constexpr size_t image_width = 4096*4;
constexpr size_t image_height = 4096*4;
constexpr size_t image_stride = image_width;

std::vector<uint8_t> generate_test_image(size_t width, size_t height, size_t stride) {
    std::vector<uint8_t> image(height * stride, 0);

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> noise(-12, 12);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const int base = static_cast<int>((x * 3 + y * 5) % 256);
            int value = base + noise(gen);

            if (((x + y) % 29) == 0) value += 80;
            if (((x * 2 + y) % 31) == 0) value -= 80;

            value = std::max(0, std::min(255, value));
            image[y * stride + x] = static_cast<uint8_t>(value);
        }
    }

    return image;
}

bool compare_images(const uint8_t* lhs, const uint8_t* rhs, size_t width, size_t height, size_t stride) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            if (lhs[y * stride + x] != rhs[y * stride + x]) return false;
        }
    }
    return true;
}

template<typename Func>
long long measure_ms(Func&& func) {
    const auto start = std::chrono::high_resolution_clock::now();
    func();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

} // namespace

int main() {
    std::cout << "2D median filter benchmark 3x3\n";
    std::cout << "Image: " << image_width << " x " << image_height << "\n";

    auto source_image = generate_test_image(image_width, image_height, image_stride);

    std::vector<uint8_t> single_thread(image_height * image_stride);
    std::vector<uint8_t> simd(image_height * image_stride);
    std::vector<uint8_t> gpu(image_height * image_stride);

    const auto single_time = measure_ms([&] {
        MedianFilter::median_filter_3x3(
            source_image.data(),
            single_thread.data(),
            image_width,
            image_height,
            image_stride
        );
    });
    std::cout << "Single thread: " << single_time << " ms\n";

    const auto simd_time = measure_ms([&] {
        MedianFilterSIMD::median_filter_3x3(
            source_image.data(),
            simd.data(),
            image_width,
            image_height,
            image_stride
        );
    });
    std::cout << "SIMD: " << simd_time << " ms\n";

    assert(compare_images(single_thread.data(), simd.data(), image_width, image_height, image_stride));
    std::cout << "CPU == SIMD: OK\n";

    if (!MedianFilterGPU::has_gpu()) {
        std::cout << "GPU device not found. GPU benchmark skipped.\n";
        return 0;
    }

    const auto gpu_time = measure_ms([&] {
        MedianFilterGPU::median_filter_3x3(
            source_image.data(),
            gpu.data(),
            image_width,
            image_height,
            image_stride
        );
    });
    std::cout << "GPU: " << gpu_time << " ms\n";

    assert(compare_images(single_thread.data(), gpu.data(), image_width, image_height, image_stride));
    std::cout << "CPU == GPU: OK\n";
    return 0;
}