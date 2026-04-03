#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <fstream>
#include <string>

#include <sycl/sycl.hpp>
#include "medianFilter.h"
#include "medianFilterSIMD.h"
#include "medianFilterGPU.h"

constexpr size_t width = 4096;
constexpr size_t height = 4096;
constexpr size_t stride = width;
constexpr float salt_pepper_probability = 0.10f;


static void write_matrix_to_file(const std::string& filename,
                                 const std::vector<uint8_t>& data,
                                 size_t width,
                                 size_t height,
                                 size_t stride,
                                 const std::string& title) {
    std::ofstream file(filename, std::ios::app);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << '\n';
        return;
    }

    file << title << "\n";
    file << "size = " << width << " x " << height << "\n";

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            file << static_cast<int>(data[y * stride + x]);
            if (x + 1 < width) file << ' ';
        }
        file << '\n';
    }

    file << "\n----------------------------------------\n\n";
}

static std::vector<uint8_t> generate_test_image(size_t w, size_t h, size_t s, float impulse_prob) {
    std::vector<uint8_t> img(h * s, 0);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> p(0.0f, 1.0f);
    std::uniform_int_distribution<int> impulse01(0, 1);

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            float base = 127.5f
                + 60.0f * std::sin(0.013f * static_cast<float>(x))
                + 45.0f * std::cos(0.017f * static_cast<float>(y));

            int value = static_cast<int>(std::lround(base));
            value = std::max(0, std::min(255, value));

            if (p(gen) < impulse_prob) {
                value = impulse01(gen) ? 255 : 0;
            }

            img[y * s + x] = static_cast<uint8_t>(value);
        }
    }

    return img;
}

static bool compare_images(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "Mismatch at index " << i
                      << ": " << static_cast<int>(a[i])
                      << " != " << static_cast<int>(b[i]) << '\n';
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "Benchmark: 2D median filter 3x3\n";
    std::cout << "Image size: " << width << " x " << height << '\n';

    std::ofstream("gpu_matrices.txt", std::ios::trunc).close();
    auto original = generate_test_image(width, height, stride, salt_pepper_probability);
    write_matrix_to_file("gpu_matrices.txt", original, width, height, stride, "Initial matrix");

    std::vector<uint8_t> out_cpu(height * stride);
    std::vector<uint8_t> out_simd(height * stride);
    std::vector<uint8_t> out_gpu(height * stride);

    auto start1 = std::chrono::high_resolution_clock::now();
    MedianFilter::median_filter_3x3(original.data(), out_cpu.data(), width, height, stride);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Single thread version: " << duration1.count() << " ms\n";

    auto start2 = std::chrono::high_resolution_clock::now();
    MedianFilterSIMD::median_filter_3x3(original.data(), out_simd.data(), width, height, stride);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "SIMD version:          " << duration2.count() << " ms\n";

    // Прогрев GPU, чтобы в замер не попадали JIT/инициализация устройства
    std::vector<uint8_t> warmup(height * stride);
    MedianFilterGPU::median_filter_3x3(original.data(), warmup.data(), width, height, stride);

    auto start3 = std::chrono::high_resolution_clock::now();
    MedianFilterGPU::median_filter_3x3(original.data(), out_gpu.data(), width, height, stride);
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    std::cout << "GPU version:           " << duration3.count() << " ms\n";
    write_matrix_to_file("gpu_matrices.txt", out_gpu, width, height, stride, "Final matrix after GPU median filter");
    

    assert(compare_images(out_cpu, out_simd));
    assert(compare_images(out_cpu, out_gpu));
    std::cout << "Results are equal!\n";

    return 0;
}