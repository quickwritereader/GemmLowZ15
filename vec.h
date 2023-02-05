
#pragma once 
#include <vecintrin.h>


constexpr int VLEN_BYTES = 16;

#define ALWAYS_INLINE __attribute__((always_inline))

template <typename T>
struct vec_inner_type_t {
    using Type __attribute__((vector_size(VLEN_BYTES))) = T;
};

template <typename T>
struct vec_type_t {
public:
    using Type = typename vec_inner_type_t<T>::Type;
    using ElementType = T;
    operator Type &() { return _val; }
    operator Type() const { return _val; }
    static constexpr int size() { return VLEN_BYTES / sizeof(ElementType); }
    ALWAYS_INLINE vec_type_t() { _val = Type {}; }

    ALWAYS_INLINE explicit vec_type_t(T scalar)
        : _val {vec_splats((T)scalar)} {}

    ALWAYS_INLINE vec_type_t(Type v) : _val {v} {}

    static vec_type_t<T> ALWAYS_INLINE loadu(const void *ptr) {
        return {vec_xl(0, reinterpret_cast<const ElementType *>(ptr))};
    }

    static ALWAYS_INLINE vec_type_t<T> loadLen(
            const void *ptr, uint32_t BYTE_INDEX) {
        return {vec_load_len(
                reinterpret_cast<const ElementType *>(ptr), BYTE_INDEX)};
    }

    static vec_type_t<T> ALWAYS_INLINE load_hinted(const void *ptr) {
        Type const *addr = (Type const *)ptr;
        Type y;
        // Doubleword aligned hint
#if __GNUC__ < 9 && !defined(__clang__)
        // hex-encode vl %[out],%[addr],3
        asm(".insn vrx,0xe70000003006,%[out],%[addr],3"
                : [out] "=v"(y)
                : [addr] "R"(*addr));
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

    vec_type_t<T> &ALWAYS_INLINE operator+=(const vec_type_t<T> &other) {
        _val = _val + other._val;
        return *this;
    }

private:
    Type _val;
};

using vuint8 = typename vec_type_t<uint8_t>::Type;
using vuint16 = typename vec_type_t<uint16_t>::Type;
using vint16 = typename vec_type_t<int16_t>::Type;
using vuint32 = typename vec_type_t<uint32_t>::Type;
using vint32 = typename vec_type_t<int32_t>::Type;

template <typename T>
std::ostream &operator<<(std::ostream &stream, const vec_type_t<T> &vec) {
    const typename vec_type_t<T>::Type v = vec;
    stream << "vec[";
    for (int i = 0; i != vec_type_t<T>::size(); i++) {
        if (i != 0) { stream << ", "; }
        stream << (typename conv_t<typename vec_type_t<T>::ElementType>::V)(
                v[i]);
    }
    stream << "]";
    return stream;
}

template < typename V, typename T>
vec_type_t<V> cast(const vec_type_t<T> &x) {
    using cast_type = typename vec_type_t<V>::Type;
    return vec_type_t<V> {(cast_type)(x.vec())};
}

// 
// const vuint16 vone16 = {1, 1, 1, 1, 1, 1, 1, 1};

inline vec_type_t<int32_t> multiplyAdd(vec_type_t<int16_t> va,
        vec_type_t<int16_t> vb, vec_type_t<int32_t> vc) { 
    // 2 ops  2 moad
    auto a = va.vec();
    auto b = vb.vec();
    auto c = vc.vec();
    c = vec_moadd(a, b, c);
    c = vec_meadd(a, b, c); 
    return vec_type_t<int32_t> {c};
}

inline vec_type_t<uint32_t> multiplySum4(vec_type_t<uint8_t> va, vec_type_t<uint8_t> vb, vec_type_t<uint32_t> vc ) {
    // 6 ops  2 mul 2 vec_sum 2 addition
    const vuint16 vz16 = {};
    const auto a = va.vec();
    const auto b = vb.vec();
    auto c = vc.vec();
    auto reso = vec_mulo(a, b);
    auto rese = vec_mule(a, b);

    c= c + vec_sum4(reso, vz16) + vec_sum4(rese, vz16); 
    return vec_type_t<uint32_t> {c};
}

inline vec_type_t<uint32_t> multiplyAdd(vec_type_t<uint8_t> va,
        vec_type_t<uint8_t> vb, vec_type_t<uint32_t> vc) { 
    // 6 ops
    vuint8 a = va.vec();
    vuint8 b = vb.vec();
    auto c = vc.vec();
    const vuint16 vone16 = {1, 1, 1, 1, 1, 1, 1, 1};
    vuint16 reso = vec_mulo(a, b);
    vuint16 rese = vec_mule(a, b);
    c = vec_moadd(reso, vone16, c);
    c = vec_meadd(reso, vone16, c);
    c = vec_moadd(rese, vone16, c);
    c = vec_meadd(rese, vone16, c); 
    return vec_type_t<uint32_t> {c};
}


 
inline vec_type_t<uint32_t> multiplySum4Low(vec_type_t<uint8_t> va, vec_type_t<uint8_t> vb, vec_type_t<uint32_t> vc) {
    // 4 ops  2 mul 1 vec_sum 1 addition
    const vuint16 vz16 = {};
    const auto a = va.vec();
    const auto b = vb.vec();
    auto c = vc.vec();
    vuint16 d = vec_mulo(a, b);
    vuint16 e = vec_meadd(a, b, d);
    c= c + vec_sum4(e, vz16);  
    return vec_type_t<uint32_t> {c};
}

