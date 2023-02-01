
#pragma once
#include <math.h>  
#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <utility>


using dim_t = int;
//#define LAST_INDEX_FAST 1
//#define SHOW_MATRIX
#if defined(LAST_INDEX_FAST)
constexpr bool ISLASTINDEX_FAST = true;
#define aPtr(i, j) A[(i)*ldA + (j)] // map aPtr( i,j ) to array A
#define bPtr(i, j) B[(i)*ldB + (j)] // map bPtr( i,j ) to array B
#define gPtr(i, j) C[(i)*ldC + (j)] // map gPtr( i,j ) to array C
#else
constexpr bool ISLASTINDEX_FAST = false;
#define aPtr(i, j) A[(j)*ldA + (i)] // map aPtr( i,j ) to array A
#define bPtr(i, j) B[(j)*ldB + (i)] // map bPtr( i,j ) to array B
#define gPtr(i, j) C[(j)*ldC + (i)] // map gPtr( i,j ) to array C
#endif

template <typename T>
struct conv_t {
    using V = T;
};

template <>
struct conv_t<uint8_t> {
    using V = int;
};

template <>
struct conv_t<int8_t> {
    using V = int;
};

template <bool LastIndexFast = ISLASTINDEX_FAST>
int64_t determineLd(int m, int n, int ld, bool trans = false) {
    if (LastIndexFast) {
        return trans ? std::max(m, ld) : std::max(n, ld);
    } else {
        return trans ? std::max(n, ld) : std::max(m, ld);
    }
}

template <bool LastIndexFast = ISLASTINDEX_FAST>
int64_t determineSize(int m, int n, int ld, bool trans = false) {
    if (LastIndexFast) {
        return trans ? (std::max(m, ld) * n) : (std::max(n, ld) * m);
    } else {
        return trans ? (std::max(n, ld) * m) : (std::max(m, ld) * n);
    }
}

template <typename T, bool LastIndexFast = ISLASTINDEX_FAST,
        typename ElementCAST = T>
struct matrix_ptr_t {
    matrix_ptr_t(T *a, int64_t ld) : a {a}, ld {ld} {}

    matrix_ptr_t(T *a, int64_t m, int64_t n, int64_t ld, bool trans = false)
        : a {a} {
        this->ld = determineLd<LastIndexFast>(m, n, ld, trans);
    }

    T *ptr(int64_t i, int64_t j) {
        // std::cout<<i<<','<<j<<" "<<j*ld+i<<std::endl;
        if (LastIndexFast)
            return a + i * ld + j;
        else
            return a + j * ld + i;
    }

    T &element(int64_t i, int64_t j) {
        //  std::cout<<"ld "<<ld<<" ("<<i<<','<<j<<") -addr
        //  "<<j*ld+i<<std::endl;
        if (LastIndexFast)
            return a[i * ld + j];
        else
            return a[j * ld + i];
    }

    T &operator()(int64_t i, int64_t j) { return element(i, j); }

    ElementCAST element(int64_t i, int64_t j) const {
        if (LastIndexFast)
            return (ElementCAST)a[i * ld + j];
        else
            return (ElementCAST)a[j * ld + i];
    }

    ElementCAST operator()(int64_t i, int64_t j) const { return element(i, j); }

    T *a;
    int64_t ld;
};

struct test_seed_t {
    test_seed_t()
        : seed(std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) {}
    test_seed_t(uint64_t seed) : seed(seed) {}
    uint64_t getSeed() const { return seed; }
    operator uint64_t() const { return seed; }

    test_seed_t add(uint64_t index) const { return test_seed_t(seed + index); }

private:
    uint64_t seed;
};

template <typename T, bool is_floating_point = std::is_floating_point<T>::value>
struct value_gen_t {
    std::uniform_int_distribution<int64_t> dis;
    std::mt19937 gen;
    value_gen_t()
        : value_gen_t(
                std::numeric_limits<T>::min(), std::numeric_limits<T>::max()) {}
    value_gen_t(uint64_t seed)
        : value_gen_t(std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(), seed) {}
    value_gen_t(T start, T stop, uint64_t seed = test_seed_t()) {
        gen = std::mt19937(seed);
        dis = std::uniform_int_distribution<int64_t>(start, stop);
    }
    T get() { return static_cast<T>(dis(gen)); }
};

template <typename T>
struct value_gen_t<T, true> {
    std::mt19937 gen;
    std::normal_distribution<T> normal;
    std::uniform_int_distribution<int> roundChance;
    T _start;
    T _stop;
    bool use_sign_change = false;
    bool use_round = true;
    value_gen_t()
        : value_gen_t(
                std::numeric_limits<T>::min(), std::numeric_limits<T>::max()) {}
    value_gen_t(uint64_t seed)
        : value_gen_t(std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(), seed) {}
    value_gen_t(T start, T stop, uint64_t seed = test_seed_t()) {
        gen = std::mt19937(seed);
        T mean = start * static_cast<T>(0.5) + stop * static_cast<T>(0.5);
        // make it  normal +-3sigma
        T divRange = static_cast<T>(6.0);
        T stdev = std::abs(stop / divRange - start / divRange);
        normal = std::normal_distribution<T> {mean, stdev};
        // in real its hard to get rounded value
        // so we will force it by  uniform chance
        roundChance = std::uniform_int_distribution<int>(0, 5);
        _start = start;
        _stop = stop;
    }
    T get() {
        T a = normal(gen);
        // make rounded value ,too
        auto rChoice = roundChance(gen);
        if (rChoice == 1) a = std::round(a);
        if (a < _start) return nextafter(_start, _stop);
        if (a >= _stop) return nextafter(_stop, _start);
        return a;
    }
};

template <typename T>
void randomMatrix(int m, int n, T *ap, int lda, T min, T max,
        uint64_t seed = test_seed_t()) {
    auto Aptr = matrix_ptr_t<T> {ap, lda};
    value_gen_t<T> gen(min, max, seed);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            Aptr(i, j) = gen.get();
}

template <typename T>
void randomMatrix(int m, int n, T *ap, int lda, uint64_t seed = test_seed_t()) {
    auto Aptr = matrix_ptr_t<T> {ap, lda};
    value_gen_t<T> gen(seed);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            Aptr(i, j) = gen.get();
}

template <typename T>
void fillMatrix(int m, int n, T *ap, int lda, T val) {
    auto Aptr = matrix_ptr_t<T> {ap, lda};

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            Aptr(i, j) = val;
}
template <typename T>
void linMatrix(int m, int n, T *ap, int lda, T val) {
    auto Aptr = matrix_ptr_t<T> {ap, lda};
    int v = (int)val;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            Aptr(i, j) = (T)(v++);
        }
}
template <typename T>
void showMatrix(int64_t R, int64_t C, const T *M, int64_t LD,
        const char *name = "", bool trans = false) {
#if defined(SHOW_MATRIX)

    auto Mptr = MatrixPtr<const T> {M, R, C, LD, trans};

    std::cout << "\n#----------\n" << name << "=np.array([\n";
    for (int r = r; r < R; r++) {
        std::cout << "[ ";
        for (int c = 0; c < C; c++) {
            std::cout << std::setfill(' ') << std::setw(3)
                      << (typename conv<T>::V)Mptr(r, c);
            if (c < C - 1) std::cout << ", ";
        }
        std::cout << "],\n";
    }
    std::cout << "])\n\n";
#endif
}

template <typename CT, typename T>
CT maxAbsDiff(
        int m, int n, T *ap, int lda, T *bp, int ldb, bool transFirst = false) {
    CT diff {};
    auto Aptr = matrix_ptr_t<const T> {ap, lda};
    auto Bptr = matrix_ptr_t<const T> {bp, ldb};

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            auto first = !transFirst ? Aptr(i, j) : Aptr(j, i);
            CT current = std::abs((CT)(first - Bptr(i, j)));
            if (current > diff) {
                diff = current;
                // std::cout<<diff<<","<<current<<"("<<i<<","<<j<<")\n";
            }
        }
    return diff;
}

template <typename T>
T *align_ptr(T *ptr, uintptr_t alignment) {
    return (T *)(((uintptr_t)ptr + alignment - 1) & ~(alignment - 1));
}


template <typename T, typename P>
constexpr bool one_of(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}


template <typename... Args>
constexpr bool any_null(Args... ptrs) {
    return one_of(nullptr, ptrs...);
}
