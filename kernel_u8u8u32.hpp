#include <stdio.h>
#include <stdlib.h>
#include <utils.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <test_reference.h>
#include <vec.h>

constexpr int fullVectored(int N, int size) {
    return (N % (VLEN_BYTES / size)) == 0;
}

constexpr int kernelType(int ROWS, int COLS, int elementSize) {
    return fullVectored(ROWS, elementSize) ? 1 : 2;
}



template <int ROWS, int COLS>
inline typename std::enable_if<kernelType(ROWS, COLS, sizeof(uint32_t)) == 1,
        void>::type
gbp(dim_t k, const uint8_t * __restrict__  MP_A, const uint8_t * __restrict__  MP_B, uint32_t * __restrict__ C, dim_t ldC) {
    using vType = typename vec_type_t<uint32_t>::Type;
    constexpr int VLEN = vec_type_t<uint32_t>::size();
    const uint32_t *MT_A = reinterpret_cast<const uint32_t *>(MP_A);
    const uint32_t *MT_B = reinterpret_cast<const uint32_t *>(MP_B);

    vec_type_t<uint32_t> Caux[ROWS / VLEN][COLS] = {};
    dim_t real_k = k / 4;
    const vuint16 vz16 = {0};
    const vuint8 mask = {0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
    if (real_k & 1) { 
            for (int i = 0; i < ROWS / VLEN; i++) {
                auto Ak = cast<uint8_t>(
                        vec_type_t<uint32_t>::load_hinted(&MT_A[i * VLEN]));

                for (int j = 0; j < COLS; j++) {
                    auto BkI = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j]});
#if !defined(LOW_0_127)
                    Caux[i][j] = multiplySum4(Ak, BkI, Caux[i][j]);
#else                    
                    Caux[i][j] = multiplySum4Low(Ak, BkI, Caux[i][j]);
#endif 
                }
            }
            MT_A += ROWS;
            MT_B += COLS;
    }

    asm("");
    dim_t real_k_2 = real_k & (-2);
    for (dim_t p = 0; p < real_k_2; p += 2) {
        for (int i = 0; i < ROWS / VLEN; i++) {
            auto Ak0 = cast<uint8_t>(vec_type_t<uint32_t>::load_hinted(&MT_A[i * VLEN]));
            auto Ak1 = cast<uint8_t>(
                    vec_type_t<uint32_t>::load_hinted(&MT_A[i * VLEN + ROWS])); 
            for (int j = 0; j < COLS; j++) {
                auto BkI0 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j]});
                auto BkI1 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j + COLS]}); 
#if !defined(LOW_0_127)
                const vuint8 a0 = Ak0;
                const vuint8 b0 = BkI0;
                const vuint8 a1 = Ak1;
                const vuint8 b1 = BkI1;
                auto reso0 = vec_mulo(a0, b0);
                auto rese0 = vec_mule(a0, b0);
                auto reso1 = vec_mulo(a1, b1);
                auto rese1 = vec_mule(a1, b1);
                auto resh = vec_perm(reso0, rese0, mask);

                Caux[i][j]  = Caux[i][j].vec() + vec_sum4(reso1, reso0)+vec_sum4(rese1, rese0) + vec_sum4(resh, vz16);
#else 
            Caux[i][j] = multiplySum4Low(Ak0, BkI0, Caux[i][j]);
            Caux[i][j] = multiplySum4Low(Ak1, BkI1, Caux[i][j]);
#endif
            }
        }

        MT_A += 2 * ROWS;
        MT_B += 2 * COLS;
    }
 
        asm("");
        for (int j = 0; j < COLS; j++) {
            for (int i = 0; i < ROWS / VLEN; i++) {
                vType *C_ij = (vType *)&gPtr(i * VLEN, j);
                *C_ij += (vType)Caux[i][j];
            }
        }

}

template <int ROWS, int COLS>
inline typename std::enable_if<kernelType(ROWS, COLS, sizeof(uint32_t)) == 2,
        void>::type
gbp(dim_t k, const uint8_t * __restrict__  MP_A, const uint8_t * __restrict__  MP_B, uint32_t * __restrict__ C, dim_t ldC) {

    using vType = typename vec_type_t<uint32_t>::Type;
    constexpr int VLEN = vec_type_t<uint32_t>::size();
    const uint32_t *MT_A = reinterpret_cast<const uint32_t *>(MP_A);
    const uint32_t *MT_B = reinterpret_cast<const uint32_t *>(MP_B);

    vec_type_t<uint32_t> Caux[COLS] = {};
    dim_t real_k = k / 4;
    dim_t real_k_2 = real_k & (-2);
    constexpr int BYTE_INDEX = ROWS * 2 * sizeof(uint8_t) - 1;
    const vuint16 vz16 = {0};
    const vuint8 mask = {0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
    if (real_k & 1) { 
            auto Ak = cast<uint8_t>(vec_type_t<uint32_t>::loadLen(MT_A, BYTE_INDEX));

            for (int j = 0; j < COLS; j++) {
                auto BkI = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j]});
#if !defined(LOW_0_127)
                Caux[j] = multiplySum4(Ak, BkI,Caux[j]);
#else
                Caux[j] = multiplySum4Low(Ak, BkI,Caux[j]);
#endif
            }

            MT_A += ROWS;
            MT_B += COLS; 
    }

    asm("");

    for (dim_t p = 0; p < real_k_2; p += 2) {

        auto Ak0 = cast<uint8_t>(vec_type_t<uint32_t>::loadLen(&MT_A[0], BYTE_INDEX));
        auto Ak1 = cast<uint8_t>(vec_type_t<uint32_t>::loadLen(&MT_A[ROWS], BYTE_INDEX));
        for (int j = 0; j < COLS; j++) {
            auto BkI0 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j]});
            auto BkI1 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j + COLS]});
#if !defined(LOW_0_127)
                const vuint8 a0 = Ak0;
                const vuint8 b0 = BkI0;
                const vuint8 a1 = Ak1;
                const vuint8 b1 = BkI1;
                auto reso0 = vec_mulo(a0, b0);
                auto rese0 = vec_mule(a0, b0);
                auto reso1 = vec_mulo(a1, b1);
                auto rese1 = vec_mule(a1, b1);
                auto resh = vec_perm(reso0, rese0, mask);

                Caux[j] = Caux[j].vec() + vec_sum4(reso1, reso0) +  vec_sum4(rese1, rese0) + vec_sum4(resh, vz16); 
#else
            Caux[j] = multiplySum4Low(Ak0, BkI0,Caux[j]);
            Caux[j] = multiplySum4Low(Ak1, BkI1,Caux[j]);

#endif
        }
        MT_A += 2 * ROWS;
        MT_B += 2 * COLS;
    }
 
        asm("");
        for (int j = 0; j < COLS; j++) {
            auto C_ij = vec_type_t<uint32_t>::loadLen(&gPtr(0, j), BYTE_INDEX);
            C_ij += Caux[j];
            C_ij.storeLen(&gPtr(0, j), BYTE_INDEX);
        } 
}

template <int M, int ROWS, int COLS>
typename std::enable_if<(M >= ROWS), void>::type LoopOne_TAIL(dim_t m, dim_t k,
     const uint8_t *Apacked, const uint8_t *Bpacked, uint32_t *C,
        dim_t ldC) {
    // end of the roll
}

template <int M, int ROWS, int COLS>
typename std::enable_if<(M < ROWS), void>::type LoopOne_TAIL(dim_t m, dim_t k,
     const uint8_t *Apacked, const uint8_t *Bpacked, uint32_t *C,
        dim_t ldC) {
    if (m & M) {
        gbp<M, COLS>(k, Apacked, Bpacked, C, ldC);
        Apacked = &Apacked[M * k];
        C = &gPtr(M, 0);
    }
    LoopOne_TAIL<2 * M, ROWS, COLS>(m, k, Apacked, Bpacked, C, ldC);
}

template <int ROWS, int COLS>
inline void LoopOne(dim_t m, dim_t k, const uint8_t *Apacked,
        const uint8_t *Bpacked, uint32_t *C, dim_t ldC) {
    for (dim_t i = 0; i < m / ROWS; i++) {
        gbp<ROWS, COLS>(k, &Apacked[i * ROWS * k],
                Bpacked, &gPtr(i * ROWS, 0), ldC);
    }
    dim_t II = m - m % ROWS;
    if (m > II)
        LoopOne_TAIL<1, ROWS, COLS>(
                m - II, k, &Apacked[II * k], Bpacked, &gPtr(II, 0), ldC);
}

template <int N, int COLS>
typename std::enable_if<(N >= COLS), void>::type LoopTwo_TAIL(dim_t m, dim_t n,
        dim_t k, const uint8_t *Apacked, const uint8_t *Bpacked, uint32_t *C,
        dim_t ldC) {
    // end of the roll
}

template <int N, int COLS>
typename std::enable_if<(N < COLS), void>::type LoopTwo_TAIL(dim_t m, dim_t n,
        dim_t k, const uint8_t *Apacked, const uint8_t *Bpacked, uint32_t *C,
        dim_t ldC) {
    if (n & N) {
        LoopOne<MR, N>(m, k, Apacked, Bpacked, C, ldC);
        Bpacked = &Bpacked[N * k];
        C = &gPtr(0, N);
    }
    LoopTwo_TAIL<2 * N, COLS>(m, n, k, Apacked, Bpacked, C, ldC);
}

template <int COLS>
__attribute__ ((noinline))  void LoopTwo(dim_t m, dim_t n, dim_t k, const uint8_t *Apacked,
        const uint8_t *Bpacked, uint32_t *C, dim_t ldC) {
    for (dim_t j = 0; j < n / COLS; j++) {
        LoopOne<MR, COLS>(m, k, Apacked, &Bpacked[j * COLS * k],
                &gPtr(0, j * COLS), ldC);
    }
    // tails , should be unrolled
    // for example n&1, n&2 and et cetera
    // actually its possible to combine tail <VLEN as
    // as we are using vec_load_len
    dim_t JJ = n - n % COLS;
    if (n > JJ)
        LoopTwo_TAIL<1, COLS>(m, n - JJ, k, Apacked, &Bpacked[JJ * k],
                &gPtr(0, JJ), ldC);
}
