
#pragma once
#include <math.h>
#include <vecintrin.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

//#define LAST_INDEX_FAST 1
//#define SHOW_MATRIX
#if defined(LAST_INDEX_FAST)
constexpr bool ISLASTINDEX_FAST = true;
#define alpha(i, j) A[(i)*ldA + (j)]  // map alpha( i,j ) to array A
#define beta(i, j) B[(i)*ldB + (j)]   // map beta( i,j ) to array B
#define gamma(i, j) C[(i)*ldC + (j)]  // map gamma( i,j ) to array C
#else
constexpr bool ISLASTINDEX_FAST = false;
#define alpha(i, j) A[(j)*ldA + (i)]  // map alpha( i,j ) to array A
#define beta(i, j) B[(j)*ldB + (i)]   // map beta( i,j ) to array B
#define gamma(i, j) C[(j)*ldC + (i)]  // map gamma( i,j ) to array C
#endif

template <typename T>
struct conv {
    using V = T;
};

template <>
struct conv<uint8_t> {
    using V = int;
};

template <>
struct conv<int8_t> {
    using V = int;
};

template <bool LastIndexFast = ISLASTINDEX_FAST>
int64_t determineLd(int m, int n, int ld, bool trans=false) {
    if (LastIndexFast) {
        return trans?std::max(m, ld)  :std::max(n, ld);
    } else {
        return trans?std::max(n, ld) :std::max(m, ld);
    }
}

template <bool LastIndexFast = ISLASTINDEX_FAST>
int64_t determineSize(int m, int n, int ld, bool trans=false) {
    if (LastIndexFast) {
        return trans?(std::max(m, ld) * n) :(std::max(n, ld) * m);
    } else {
        return trans?(std::max(n, ld) * m):(std::max(m, ld) * n);
    }
}

template <typename T, bool LastIndexFast = ISLASTINDEX_FAST,
          typename ElementCAST = T>
struct MatrixPtr {
    MatrixPtr(T *a, int64_t ld) : a{a}, ld{ld} {}

    MatrixPtr(T *a, int64_t m, int64_t n, int64_t ld, bool trans=false) : a{a} {
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

struct TestSeed {
    TestSeed()
        : seed(std::chrono::high_resolution_clock::now()
                   .time_since_epoch()
                   .count()) {}
    TestSeed(uint64_t seed) : seed(seed) {}
    uint64_t getSeed() { return seed; }
    operator uint64_t() const { return seed; }

    TestSeed add(uint64_t index) { return TestSeed(seed + index); }

   private:
    uint64_t seed;
};

template <typename T, bool is_floating_point = std::is_floating_point<T>::value>
struct ValueGen {
    std::uniform_int_distribution<int64_t> dis;
    std::mt19937 gen;
    ValueGen()
        : ValueGen(std::numeric_limits<T>::min(),
                   std::numeric_limits<T>::max()) {}
    ValueGen(uint64_t seed)
        : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(),
                   seed) {}
    ValueGen(T start, T stop, uint64_t seed = TestSeed()) {
        gen = std::mt19937(seed);
        dis = std::uniform_int_distribution<int64_t>(start, stop);
    }
    T get() { return static_cast<T>(dis(gen)); }
};

template <typename T>
struct ValueGen<T, true> {
    std::mt19937 gen;
    std::normal_distribution<T> normal;
    std::uniform_int_distribution<int> roundChance;
    T _start;
    T _stop;
    bool use_sign_change = false;
    bool use_round = true;
    ValueGen()
        : ValueGen(std::numeric_limits<T>::min(),
                   std::numeric_limits<T>::max()) {}
    ValueGen(uint64_t seed)
        : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(),
                   seed) {}
    ValueGen(T start, T stop, uint64_t seed = TestSeed()) {
        gen = std::mt19937(seed);
        T mean = start * static_cast<T>(0.5) + stop * static_cast<T>(0.5);
        // make it  normal +-3sigma
        T divRange = static_cast<T>(6.0);
        T stdev = std::abs(stop / divRange - start / divRange);
        normal = std::normal_distribution<T>{mean, stdev};
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
                  uint64_t seed = TestSeed()) {
    auto Aptr = MatrixPtr<T>{ap, lda};
    ValueGen<T> gen(min, max, seed);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) Aptr(i, j) = gen.get();
}

template <typename T>
void randomMatrix(int m, int n, T *ap, int lda, uint64_t seed = TestSeed()) {
    auto Aptr = MatrixPtr<T>{ap, lda};
    ValueGen<T> gen(seed);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) Aptr(i, j) = gen.get();
}

template <typename T>
void fillMatrix(int m, int n, T *ap, int lda, T val) {
    auto Aptr = MatrixPtr<T>{ap, lda};

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) Aptr(i, j) = val;
}
template <typename T>
void linMatrix(int m, int n, T *ap, int lda, T val) {
    auto Aptr = MatrixPtr<T>{ap, lda};
    int v = (int)val;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            Aptr(i, j) = (T)(v++);
        }
}
template <typename T>
void showMatrix(int64_t R, int64_t C, const T *M, int64_t LD, 
                const char *name = "",bool trans = false) {
#if defined(SHOW_MATRIX)

    auto Mptr = MatrixPtr<const T>{M, R, C, LD, trans};

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
CT maxAbsDiff(int m, int n, T *ap, int lda, T *bp, int ldb,
              bool transFirst = false) {
    CT diff{};
    auto Aptr = MatrixPtr<const T>{ap, lda};
    auto Bptr = MatrixPtr<const T>{bp, ldb};

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

constexpr int VLEN_BYTES = 16;

#define ALWAYS_INLINE __attribute__((always_inline))

template <typename T>
struct VecInnerType {
    using Type __attribute__((vector_size(VLEN_BYTES))) = T;
};

template <typename T>
struct VecType {
   public:
    using Type = typename VecInnerType<T>::Type;
    using ElementType = T;
    operator Type &() { return _val; }
    operator Type() const { return _val; }
    static constexpr int size() { return VLEN_BYTES / sizeof(ElementType); }
    ALWAYS_INLINE VecType() { _val = Type{};}

    ALWAYS_INLINE explicit VecType(T scalar) : _val{vec_splats((T)scalar)} {}

    ALWAYS_INLINE VecType(Type v) : _val{v} {}

    static VecType<T> ALWAYS_INLINE loadu(const void *ptr) {
        return {vec_xl(0, reinterpret_cast<const ElementType *>(ptr))};
    }

    static ALWAYS_INLINE VecType<T> loadLen(const void *ptr,
                                            uint32_t BYTE_INDEX) {
        return {vec_load_len(reinterpret_cast<const ElementType *>(ptr),
                             BYTE_INDEX)};
    }

    static VecType<T> ALWAYS_INLINE load_hinted(const void *ptr) {
        Type const *addr = (Type const *)ptr;
        Type y;
        // Doubleword aligned hint
#if __GNUC__ < 9 && !defined(__clang__)
        // hex-encode vl %[out],%[addr],3
        asm(".insn vrx,0xe70000003006,%[out],%[addr],3"
            : [ out ] "=v"(y)
            : [ addr ] "R"(*addr));
#else
        y = *addr;
#endif

        return y;
    }

    void ALWAYS_INLINE store(void *ptr) const {
        vec_xst(_val, 0, reinterpret_cast<ElementType *>(ptr));
    }

    void ALWAYS_INLINE storeLen(void *ptr, uint32_t BYTE_INDEX) const {
        vec_store_len(_val, reinterpret_cast<ElementType *>(ptr), BYTE_INDEX);
    }
    ALWAYS_INLINE const Type &vec() const { return _val; }

    VecType<T> &ALWAYS_INLINE operator+=(const VecType<T> &other) {
        _val = _val + other._val;
        return *this;
    }

   private:
    Type _val;
};

using vuint8 = typename VecType<uint8_t>::Type;
using vuint16 = typename VecType<uint16_t>::Type;
using vuint32 = typename VecType<uint32_t>::Type;

template <typename T>
std::ostream &operator<<(std::ostream &stream, const VecType<T> &vec) {
    const typename VecType<T>::Type v = vec;
    stream << "vec[";
    for (int i = 0; i != VecType<T>::size(); i++) {
        if (i != 0) {
            stream << ", ";
        }
        stream << (typename conv<typename VecType<T>::ElementType>::V)(v[i]);
    }
    stream << "]";
    return stream;
}

template <typename T, typename V>
VecType<V> cast(const VecType<T> &x) {
    using cast_type = typename VecType<V>::Type;
    return VecType<V>{(cast_type)(x.vec())};
}

const vuint16 vz16 = {0};
const vuint16 vone16 = {1, 1, 1, 1, 1, 1, 1, 1};

inline vuint32 multiplySum4(VecType<uint8_t> va, VecType<uint8_t> vb) {
    //    std::cout<<va<<std::endl;
    //    std::cout<<vb<<std::endl;
    const vuint8 a = va;
    const vuint8 b = vb;
    auto reso = vec_mulo(a, b);
    auto rese = vec_mule(a, b);

    auto ret = vec_sum4(reso, vz16);
    ret += vec_sum4(rese, vz16);
    // asm("");
    // std::cout<<"ret: "<<VecType<uint32_t>{ret}<<std::endl;
    return ret;
}

inline vuint32 multiplySum4_mulo(VecType<uint8_t> va, VecType<uint8_t> vb) {
    //  std::cout<<va<<std::endl;
    //  std::cout<<vb<<std::endl;
    const vuint8 a = va;
    const vuint8 b = vb;
    auto reso = vec_mulo(a, b);

    return vec_sum4(reso, vz16);
}

inline vuint32 multiplySum4_mule(VecType<uint8_t> va, VecType<uint8_t> vb) {
    const vuint8 a = va;
    const vuint8 b = vb;
    auto rese = vec_mule(a, b);

    return vec_sum4(rese, vz16);
}

inline vuint32 multiplySum4Low(VecType<uint8_t> va, VecType<uint8_t> vb) {
    //  std::cout<<va<<std::endl;
    //  std::cout<<vb<<std::endl;
    const vuint8 a = va;
    const vuint8 b = vb;
    vuint16 d = vec_moadd(a, b, vz16);
    vuint16 e = vec_meadd(a, b, d);
    auto ret = vec_sum4(e, vz16);  // + vec_sum4(rese, vz16);

    // std::cout<<VecType<uint32_t>{ret}<<std::endl;
    return ret;
}


inline VecType<uint32_t> multiplyAdd(VecType<uint16_t> va, VecType<uint16_t> vb , VecType<uint32_t> vc) {
//  std::cout<<"-----\n"; 
//      std::cout<<va<<std::endl;
//      std::cout<<vb<<std::endl;
//      std::cout<<vc<<std::endl;
    const vuint16 a = va;
    const vuint16 b = vb;
    vuint32 c = vc;
    c = vec_moadd(a, b, c);
    c = vec_meadd(a, b, c); 
// std::cout<<"=\n"; 
    // std::cout<<VecType<uint32_t>{c}<<std::endl;
    return VecType<uint32_t>{c};
}