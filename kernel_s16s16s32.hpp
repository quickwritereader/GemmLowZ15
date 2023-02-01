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
inline typename std::enable_if<kernelType(ROWS, COLS, sizeof(int32_t)) == 1,
        void>::type
gbp(dim_t k, const int16_t * __restrict__  MP_A, const int16_t * __restrict__  MP_B, int32_t * __restrict__ C, dim_t ldC) {
    using vType = typename vec_type_t<int32_t>::Type;
    constexpr int VLEN = vec_type_t<int32_t>::size();
    const int32_t *MT_A = reinterpret_cast<const int32_t *>(MP_A);
    const int32_t *MT_B = reinterpret_cast<const int32_t *>(MP_B);

    vec_type_t<int32_t> Caux[ROWS / VLEN][COLS] = {};
    dim_t real_k = k / 2;
    dim_t real_k_4 = real_k & (-4);

    if (real_k & 3) {
        for (dim_t p = 0; p < (real_k & 3); p++) {
            for (int i = 0; i < ROWS / VLEN; i++) {
                auto Ak = cast<int16_t>(
                        vec_type_t<int32_t>::load_hinted(&MT_A[i * VLEN]));

                for (int j = 0; j < COLS; j++) {
                    auto BkI = cast<int16_t>(vec_type_t<int32_t> {MT_B[j]});

                    Caux[i][j] = multiplyAdd(Ak, BkI, Caux[i][j]);
                }
            }
            MT_A += ROWS;
            MT_B += COLS;
        }
    }

    asm("");

    for (dim_t p = 0; p < real_k_4; p += 4) {
        for (int i = 0; i < ROWS / VLEN; i++) {
            auto Ak0 = cast<int16_t>(vec_type_t<int32_t>::load_hinted(&MT_A[i * VLEN]));
            auto Ak1 = cast<int16_t>(
                    vec_type_t<int32_t>::load_hinted(&MT_A[i * VLEN + ROWS]));
            auto Ak2 = cast<int16_t>(
                    vec_type_t<int32_t>::load_hinted(&MT_A[i * VLEN + 2 * ROWS]));
            auto Ak3 = cast<int16_t>(
                    vec_type_t<int32_t>::load_hinted(&MT_A[i * VLEN + 3 * ROWS]));
            for (int j = 0; j < COLS; j++) {
                auto BkI0 = cast<int16_t>(vec_type_t<int32_t> {MT_B[j]});
                auto BkI1 = cast<int16_t>(vec_type_t<int32_t> {MT_B[j + COLS]});
                auto BkI2 = cast<int16_t>(vec_type_t<int32_t> {MT_B[j + 2 * COLS]});
                auto BkI3 = cast<int16_t>(vec_type_t<int32_t> {MT_B[j + 3 * COLS]});

                Caux[i][j] = multiplyAdd(Ak0, BkI0, Caux[i][j]);
                Caux[i][j] = multiplyAdd(Ak1, BkI1, Caux[i][j]);
                Caux[i][j] = multiplyAdd(Ak2, BkI2, Caux[i][j]);
                Caux[i][j] = multiplyAdd(Ak3, BkI3, Caux[i][j]);
            }
        }

        MT_A += 4 * ROWS;
        MT_B += 4 * COLS;
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
inline typename std::enable_if<kernelType(ROWS, COLS, sizeof(int32_t)) == 2,
        void>::type
gbp(dim_t k, const int16_t * __restrict__  MP_A, const int16_t * __restrict__  MP_B, int32_t * __restrict__  C, dim_t ldC) {

    using vType = typename vec_type_t<int32_t>::Type;
    constexpr int VLEN = vec_type_t<int32_t>::size();
    const int32_t *MT_A = reinterpret_cast<const int32_t *>(MP_A);
    const int32_t *MT_B = reinterpret_cast<const int32_t *>(MP_B);

    vec_type_t<int32_t> Caux[COLS] = {};
    dim_t real_k = k / 2;
    dim_t real_k_4 = real_k & (-4);
    constexpr int BYTE_INDEX = ROWS * 2 * sizeof(int16_t) - 1;

    if (real_k & 3) {
        for (dim_t p = 0; p < (real_k & 3); p++) {

            auto Ak = cast<int16_t>(vec_type_t<int32_t>::loadLen(MT_A, BYTE_INDEX));

            for (int j = 0; j < COLS; j++) {
                auto BkI = cast<int16_t>(vec_type_t<int32_t> {MT_B[j]});

                Caux[j] = multiplyAdd(Ak, BkI, Caux[j]);
            }

            MT_A += ROWS;
            MT_B += COLS;
        }
    }

    asm("");

    for (dim_t p = 0; p < real_k_4; p += 4) {

        auto Ak0 = cast<int16_t>(vec_type_t<int32_t>::loadLen(&MT_A[0], BYTE_INDEX));
        auto Ak1 = cast<int16_t>(vec_type_t<int32_t>::loadLen(&MT_A[ROWS], BYTE_INDEX));
        auto Ak2 = cast<int16_t>(
                vec_type_t<int32_t>::loadLen(&MT_A[2 * ROWS], BYTE_INDEX));
        auto Ak3 = cast<int16_t>(
                vec_type_t<int32_t>::loadLen(&MT_A[3 * ROWS], BYTE_INDEX));
        for (int j = 0; j < COLS; j++) {
            auto BkI0 = cast<int16_t>(vec_type_t<int32_t> {MT_B[j]});
            auto BkI1 = cast<int16_t>(vec_type_t<int32_t> {MT_B[j + COLS]});
            auto BkI2 = cast<int16_t>(vec_type_t<int32_t> {MT_B[j + 2 * COLS]});
            auto BkI3 = cast<int16_t>(vec_type_t<int32_t> {MT_B[j + 3 * COLS]});

            Caux[j] = multiplyAdd(Ak0, BkI0, Caux[j]);
            Caux[j] = multiplyAdd(Ak1, BkI1, Caux[j]);
            Caux[j] = multiplyAdd(Ak2, BkI2, Caux[j]);
            Caux[j] = multiplyAdd(Ak3, BkI3, Caux[j]);
        }
        MT_A += 4 * ROWS;
        MT_B += 4 * COLS;
    }
 
        asm("");
        for (int j = 0; j < COLS; j++) {
            auto C_ij = vec_type_t<int32_t>::loadLen(&gPtr(0, j), BYTE_INDEX);
            C_ij += Caux[j];
            C_ij.storeLen(&gPtr(0, j), BYTE_INDEX);
        } 
}

template <int M, int ROWS, int COLS>
typename std::enable_if<(M >= ROWS), void>::type LoopOne_TAIL(dim_t m, dim_t k,
     const int16_t *Apacked, const int16_t *Bpacked, int32_t *C,
        dim_t ldC) {
    // end of the roll
}

template <int M, int ROWS, int COLS>
typename std::enable_if<(M < ROWS), void>::type LoopOne_TAIL(dim_t m, dim_t k,
     const int16_t *Apacked, const int16_t *Bpacked, int32_t *C,
        dim_t ldC) {
    if (m & M) {
        gbp<M, COLS>(k, Apacked, Bpacked, C, ldC);
        Apacked = &Apacked[M * k];
        C = &gPtr(M, 0);
    }
    LoopOne_TAIL<2 * M, ROWS, COLS>(m, k, Apacked, Bpacked, C, ldC);
}

template <int ROWS, int COLS>
inline void LoopOne(dim_t m, dim_t k, const int16_t *Apacked,
        const int16_t *Bpacked, int32_t *C, dim_t ldC) {
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
        dim_t k, const int16_t *Apacked, const int16_t *Bpacked, int32_t *C,
        dim_t ldC) {
    // end of the roll
}

template <int N, int COLS>
typename std::enable_if<(N < COLS), void>::type LoopTwo_TAIL(dim_t m, dim_t n,
        dim_t k, const int16_t *Apacked, const int16_t *Bpacked, int32_t *C,
        dim_t ldC) {
    if (n & N) {
        LoopOne<MR, N>(m, k, Apacked, Bpacked, C, ldC);
        Bpacked = &Bpacked[N * k];
        C = &gPtr(0, N);
    }
    LoopTwo_TAIL<2 * N, COLS>(m, n, k, Apacked, Bpacked, C, ldC);
}

template <int COLS>
void LoopTwo(dim_t m, dim_t n, dim_t k, const int16_t *Apacked,
        const int16_t *Bpacked, int32_t *C, dim_t ldC) {
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
