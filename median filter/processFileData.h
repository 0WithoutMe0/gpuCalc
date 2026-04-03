#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <iomanip>


std::vector<float> generate_test_data(size_t size, float noise_level, float outlier, size_t outlier_step);
void write_array_to_file(const char* filename, const float* input_data, const float* filtered_data, size_t size);
bool compare_data(const float* A, const float* B, size_t size);


//генерация тестовых данных
std::vector<float> generate_test_data(size_t size, float noise_level, float outlier, size_t outlier_step) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-noise_level, noise_level);

    //сигнал: синус + шум
    for (size_t i = 0; i < size; ++i) {
        float signal = std::sin(i * 0.1f);
        float noise = dist(gen);
        //добавляем выбросы
        if (outlier_step > 0 && i % outlier_step == 0 && i > 0) {
            float direction = (signal >= 0) ? 1.0f : -1.0f;
            noise += outlier * direction;
        }
        data[i] = signal + noise;
    }

    return data;
}

//запись в CSV-файл в формате [input; filtered]
void write_array_to_file(const char* filename, const float* input_data, const float* filtered_data, size_t size) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "input;filtered\n";
    for (size_t i = 0; i < size; ++i) {
        file << std::fixed << std::setprecision(3) << input_data[i] << ";"
            << std::fixed << std::setprecision(3) << filtered_data[i] << "\n";
    }

    std::cout << "Data file complete" << std::endl;
}

//сравнение результатов 2-ух фильтров
bool compare_data(const float* A, const float* B, size_t size) {
    for (size_t i = 0; i < size; ++i)
        if (A[i] != B[i]) return false;
    return true;
}