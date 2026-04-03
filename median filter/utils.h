#pragma once

template<typename T>
static T get_min(T a, T b) { return a < b ? a : b; }

template<typename T>
static T get_max(T a, T b) { return a > b ? a : b; }

//компаратор сортирующей сети
template<typename T>
static void cond_swap(T& a, T& b) {
    if (a > b) {
        T tmp = a;
        a = b;
        b = tmp;
    }
}
