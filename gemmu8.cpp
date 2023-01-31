#include <stdio.h>
#include <stdlib.h>
#include <utils.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <pack.hpp>

// constexpr int MR=12;
// constexpr int NR=4;
constexpr int MC = 48 * 8 * 2;
constexpr int KC = 164 * 4 * 2;
constexpr int NC = 2000;
void LoopFive(bool transA, bool transB, int m, int n, int k, uint8_t *A,
        int ldA, uint8_t *B, int ldB, uint32_t *C, int ldC);
void LoopFour(bool transA, bool transB, int m, int n, int k, uint8_t *A,
        int ldA, uint8_t *B, int ldB, uint32_t *C, int ldC, uint8_t *Apacked,
        uint8_t *Bpacked);
void LoopThree(bool transA, bool transB, int m, int n, int k, uint8_t *A,
        int ldA, uint8_t *Bpacked, uint32_t *C, int ldC, uint8_t *Apacked);
template <int COLS>
void LoopTwo(int m, int n, int k, uint8_t *Apacked, uint8_t *Bpacked,
        uint32_t *C, int ldC);
template <int ROWS, int COLS>
void LoopOne(
        int m, int k, uint8_t *Apacked, uint8_t *Bpacked, uint32_t *C, int ldC);

void MyGemm(bool transA, bool transB, int m, int n, int k, uint8_t *A, int ldA,
        uint8_t *B, int ldB, uint32_t *C, int ldC) {
    LoopFive(transA, transB, m, n, k, A, ldA, B, ldB, C, ldC);
}

inline void LoopFive(bool transA, bool transB, int m, int n, int k, uint8_t *A,
        int ldA, uint8_t *B, int ldB, uint32_t *C, int ldC) {
    std::unique_ptr<uint8_t[]> Bpack(new uint8_t[KC * NC]);
    std::unique_ptr<uint8_t[]> Apack(new uint8_t[MC * KC]);
    for (int j = 0; j < n; j += NC) {
        int jb = std::min(
                NC, n - j); /* Last loop may not involve a full block */
        LoopFour(transA, transB, m, jb, k, A, ldA,
                transB ? &beta(j, 0) : &beta(0, j), ldB, &gamma(0, j), ldC,
                Apack.get(), Bpack.get());
    }
}

inline void LoopFour(bool transA, bool transB, int m, int n, int k, uint8_t *A,
        int ldA, uint8_t *B, int ldB, uint32_t *C, int ldC, uint8_t *Apacked,
        uint8_t *Bpacked) {
    constexpr int VLEN = VecType<uint32_t>::size();
    for (int p = 0; p < k; p += KC) {
        int pb = std::min(KC, k - p);
        if (transB) {
            pack_K<uint8_t, uint8_t, NR, 4, true>(
                    pb, n, &beta(0, p), ldB, Bpacked);
        } else {
            pack_K<uint8_t, uint8_t, NR, 4, false>(
                    pb, n, &beta(p, 0), ldB, Bpacked);
        }
        showMatrix(4, ((pb + 3) & (-4)) * n / 4, Bpacked, 1, "Bpack");
        LoopThree(transA, transB, m, n, pb,
                transA ? &alpha(p, 0) : &alpha(0, p), ldA, Bpacked, C, ldC,
                Apacked);
    }
}
inline void LoopThree(bool transA, bool transB, int m, int n, int k, uint8_t *A,
        int ldA, uint8_t *Bpacked, uint32_t *C, int ldC, uint8_t *Apacked) {
    constexpr int VLEN = VecType<uint32_t>::size();
    for (int i = 0; i < m; i += MC) {
        int ib = std::min(
                MC, m - i); /* Last loop may not involve a full block */
        if (transA) {
            pack_K<uint8_t, uint8_t, MR, 4, false>(
                    k, ib, &alpha(0, i), ldA, Apacked);
        } else {
            pack_K<uint8_t, uint8_t, MR, 4, true>(
                    k, ib, &alpha(i, 0), ldA, Apacked);
        }

        showMatrix(((k + 3) & (-4)) * ib / 4, 4, Apacked, 1, "Apack");
        int kk = (k + 3) & -4;
        LoopTwo<NR>(ib, n, kk, Apacked, Bpacked, &gamma(i, 0), ldC);
    }
}

constexpr int fullVectored(int N, int size) {
    return (N % (VLEN_BYTES / size)) == 0;
}

constexpr int kernelType(int ROWS, int COLS, int elementSize) {
    return fullVectored(ROWS, elementSize) ? 1 : 2;
}

template <int ROWS, int COLS, typename T, typename DT>
inline typename std::enable_if<kernelType(ROWS, COLS, sizeof(T)) == 1,
        void>::type
gbp(int k, const DT *MP_A, const DT *MP_B, T *C, int ldC) {
    using vType = typename VecType<T>::Type;
    using elType = typename VecType<T>::ElementType;
    constexpr int VLEN = VecType<T>::size();
    constexpr int VLEN_DT = VecType<DT>::size();

    vType Caux[ROWS / VLEN][COLS] = {};
    const T *MT_A = reinterpret_cast<const T *>(MP_A);
    const T *MT_B = reinterpret_cast<const T *>(MP_B);
    int real_k = k / 4;
    int real_k_2 = real_k & (-2);

    if (real_k & 1) {
        for (int i = 0; i < ROWS / VLEN; i++) {
            auto Ak = cast<T, DT>(VecType<T>::loadu(&MT_A[i * VLEN]));

            for (int j = 0; j < COLS; j++) {
                auto BkI = cast<T, DT>(VecType<T> {MT_B[j]});

                Caux[i][j] += multiplySum4(Ak, BkI);
            }
        }
        MT_A += ROWS;
        MT_B += COLS;
    }

    asm("");
    vuint8 mask = {0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
    vuint16 vz16 = {};
    for (int p = 0; p < real_k_2; p += 2) {
        // std::cout<<p<<"  "<<VecType<T>{Caux[0][0]}<<std::endl;
        for (int i = 0; i < ROWS / VLEN; i++) {
            auto Ak0 = cast<T, DT>(VecType<T>::load_hinted(&MT_A[i * VLEN]));
            auto Ak1 = cast<T, DT>(
                    VecType<T>::load_hinted(&MT_A[i * VLEN + ROWS]));
            for (int j = 0; j < COLS; j++) {
                auto BkI0 = cast<T, DT>(VecType<T> {MT_B[j]});
                auto BkI1 = cast<T, DT>(VecType<T> {MT_B[j + COLS]});
                // Caux[i][j] += multiplySum4(Ak0, BkI0);
                // Caux[i][j] += multiplySum4(Ak1, BkI1);
                const vuint8 a = Ak0;
                const vuint8 b = BkI0;
                const vuint8 a1 = Ak1;
                const vuint8 b1 = BkI1;
                auto reso0 = vec_mulo(a, b);
                auto rese0 = vec_mule(a, b);
                auto reso1 = vec_mulo(a1, b1);
                auto rese1 = vec_mule(a1, b1);
                auto resh = vec_perm(reso0, rese0, mask);

                Caux[i][j] += vec_sum4(reso1, reso0);
                Caux[i][j] += vec_sum4(rese1, rese0);
                Caux[i][j] += vec_sum4(resh, vz16);
            }
        }

        MT_A += 2 * ROWS;
        MT_B += 2 * COLS;
    }

    asm("");
#if !defined(LAST_INDEX_FAST)

    for (int j = 0; j < COLS; j++) {
        for (int i = 0; i < ROWS / VLEN; i++) {
            vType *C_ij = (vType *)&gamma(i * VLEN, j);
            *C_ij += Caux[i][j];
        }
    }
#else
#pragma GCC error "was not implemented"
#endif
}

template <int ROWS, int COLS, typename T, typename DT>
inline typename std::enable_if<kernelType(ROWS, COLS, sizeof(T)) == 2,
        void>::type
gbp(int k, const DT *MP_A, const DT *MP_B, T *C, int ldC) {
    using vType = typename VecType<T>::Type;
    using elType = typename VecType<T>::ElementType;
    constexpr int VLEN = VecType<T>::size();
    constexpr int VLEN_DT = VecType<DT>::size();

    // we gonna use vector still, but this time
    // ROWS will be 1 with vector loaded by length
    vType Caux[COLS] = {};
    const T *MT_A = reinterpret_cast<const T *>(MP_A);
    const T *MT_B = reinterpret_cast<const T *>(MP_B);
    int real_k = k / 4;
    int real_k_2 = real_k & (-2);
    constexpr int BYTE_INDEX = ROWS * 4 - 1; // 4 k packed

    if (real_k & 1) {
        auto Ak = cast<T, DT>(VecType<T>::loadLen(MT_A, BYTE_INDEX));

        for (int j = 0; j < COLS; j++) {
            auto BkI = cast<T, DT>(VecType<T> {MT_B[j]});

            Caux[j] += multiplySum4(Ak, BkI);
        }
        MT_A += ROWS;
        MT_B += COLS;
    }

    asm("");
    vuint8 mask = {0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
    vuint16 vz16 = {};
    for (int p = 0; p < real_k_2; p += 2) {
        auto Ak0 = cast<T, DT>(VecType<T>::loadLen(MT_A, BYTE_INDEX));
        auto Ak1 = cast<T, DT>(VecType<T>::loadLen(&MT_A[ROWS], BYTE_INDEX));
        for (int j = 0; j < COLS; j++) {
            auto BkI0 = cast<T, DT>(VecType<T> {MT_B[j]});
            auto BkI1 = cast<T, DT>(VecType<T> {MT_B[j + COLS]});
            // Caux[j] += multiplySum4(Ak0, BkI0);
            // Caux[j] += multiplySum4(Ak1, BkI1);
            const vuint8 a = Ak0;
            const vuint8 b = BkI0;
            const vuint8 a1 = Ak1;
            const vuint8 b1 = BkI1;
            auto reso0 = vec_mulo(a, b);
            auto rese0 = vec_mule(a, b);
            auto reso1 = vec_mulo(a1, b1);
            auto rese1 = vec_mule(a1, b1);
            auto resh = vec_perm(reso0, rese0, mask);

            Caux[j] += vec_sum4(reso1, reso0);
            Caux[j] += vec_sum4(rese1, rese0);
            Caux[j] += vec_sum4(resh, vz16);
        }

        MT_A += 2 * ROWS;
        MT_B += 2 * COLS;
    }

    asm("");
#if !defined(LAST_INDEX_FAST)

    for (int j = 0; j < COLS; j++) {
        auto C_ij = VecType<T>::loadLen(&gamma(0, j), BYTE_INDEX);
        C_ij += Caux[j];
        C_ij.storeLen(&gamma(0, j), BYTE_INDEX);
    }
#else
#pragma GCC error "was not implemented"
#endif
}

template <int M, int ROWS, int COLS>
typename std::enable_if<(M >= ROWS), void>::type LoopOne_TAIL(int m, int k,
        uint8_t *Apacked, uint8_t *Bpacked, uint32_t *C, int ldC) {
    // end of the roll
}

template <int M, int ROWS, int COLS>
typename std::enable_if<(M < ROWS), void>::type LoopOne_TAIL(int m, int k,
        uint8_t *Apacked, uint8_t *Bpacked, uint32_t *C, int ldC) {
    if (m & M) {
        gbp<M, COLS, uint32_t, uint8_t>(k, Apacked, Bpacked, C, ldC);
        Apacked = &Apacked[M * k];
        C = &gamma(M, 0);
        // std::cout<<C<<std::endl;;
    }
    LoopOne_TAIL<2 * M, ROWS, COLS>(m, k, Apacked, Bpacked, C, ldC);
}

template <int ROWS, int COLS>
inline void LoopOne(int m, int k, uint8_t *Apacked, uint8_t *Bpacked,
        uint32_t *C, int ldC) {
    for (int i = 0; i < m / ROWS; i++) {
        gbp<ROWS, COLS, uint32_t, uint8_t>(
                k, &Apacked[i * ROWS * k], Bpacked, &gamma(i * ROWS, 0), ldC);
    }
    int II = m - m % ROWS;
    if (m > II)
        LoopOne_TAIL<1, ROWS, COLS>(
                m - II, k, &Apacked[II * k], Bpacked, &gamma(II, 0), ldC);
}

template <int N, int COLS>
typename std::enable_if<(N >= COLS), void>::type LoopTwo_TAIL(int m, int n,
        int k, uint8_t *Apacked, uint8_t *Bpacked, uint32_t *C, int ldC) {
    // end of the roll
}

template <int N, int COLS>
typename std::enable_if<(N < COLS), void>::type LoopTwo_TAIL(int m, int n,
        int k, uint8_t *Apacked, uint8_t *Bpacked, uint32_t *C, int ldC) {
    if (n & N) {
        LoopOne<MR, N>(m, k, Apacked, Bpacked, C, ldC);
        Bpacked = &Bpacked[N * k];
        C = &gamma(0, N);
    }
    LoopTwo_TAIL<2 * N, COLS>(m, n, k, Apacked, Bpacked, C, ldC);
}

template <int COLS>
inline void LoopTwo(int m, int n, int k, uint8_t *Apacked, uint8_t *Bpacked,
        uint32_t *C, int ldC) {
    for (int j = 0; j < n / COLS; j++) {
        LoopOne<MR, COLS>(m, k, Apacked, &Bpacked[j * COLS * k],
                &gamma(0, j * COLS), ldC);
    }
    // tails , should be unrolled
    // for example n&1, n&2 and et cetera
    // actually its possible to combine tail <VLEN as
    // as we are using vec_load_len
    int JJ = n - n % COLS;
    if (n > JJ)
        LoopTwo_TAIL<1, COLS>(
                m, n - JJ, k, Apacked, &Bpacked[JJ * k], &gamma(0, JJ), ldC);
}
