#include <stdio.h>
#include <stdlib.h>
#include <utils.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <pack.hpp>

// constexpr int MR=12;
// constexpr int NR=4;
constexpr int MC = 48 * 8 *2 ;
constexpr int KC = 164 * 4;
constexpr int NC = 2000;
void LoopFive(int m, int n, int k, uint8_t *A, int ldA, uint8_t *B, int ldB,
              uint32_t *C, int ldC);
void LoopFour(int m, int n, int k, uint8_t *A, int ldA, uint8_t *B, int ldB,
              uint32_t *C, int ldC, uint16_t *Apacked, uint16_t *Bpacked);
void LoopThree(int m, int n, int k, uint8_t *A, int ldA, uint16_t *Bpacked,
               uint32_t *C, int ldC, uint16_t *Apacked);
template <int COLS>
void LoopTwo(int m, int n, int k, uint16_t *Apacked, uint16_t *Bpacked, uint32_t *C,
             int ldC);
template <int ROWS, int COLS>
void LoopOne(int m, int k, uint16_t *Apacked, uint16_t *Bpacked, uint32_t *C,
             int ldC);

void MyGemm(int m, int n, int k, uint8_t *A, int ldA, uint8_t *B, int ldB,
            uint32_t *C, int ldC) {
    LoopFive(m, n, k, A, ldA, B, ldB, C, ldC);
}

inline void LoopFive(int m, int n, int k, uint8_t *A, int ldA, uint8_t *B,
                     int ldB, uint32_t *C, int ldC) {
    std::unique_ptr<uint16_t[]> Bpack(new uint16_t[KC * NC]);
    std::unique_ptr<uint16_t[]> Apack(new uint16_t[MC * KC]);
    for (int j = 0; j < n; j += NC) {
        int jb =
            std::min(NC, n - j); /* Last loop may not involve a full block */
        LoopFour(m, jb, k, A, ldA, &beta(0, j), ldB, &gamma(0, j), ldC,
                 Apack.get(), Bpack.get());
    }
}

inline void LoopFour(int m, int n, int k, uint8_t *A, int ldA, uint8_t *B,
                     int ldB, uint32_t *C, int ldC, uint16_t *Apacked,
                     uint16_t *Bpacked) {
    constexpr int VLEN = VecType<uint32_t>::size();
    for (int p = 0; p < k; p += KC) {
        int pb = std::min(KC, k - p);

        pack_K<uint8_t, uint16_t, NR, 2, false>(pb, n, &beta(p, 0), ldB, Bpacked);
        showMatrix(2, ((pb+1)&(-2)) * n/2 , Bpacked, 1, "Bpack");

        LoopThree(m, n, pb, &alpha(0, p), ldA, Bpacked, C, ldC, Apacked);
    }
}

inline void LoopThree(int m, int n, int k, uint8_t *A, int ldA, uint16_t *Bpacked,
                      uint32_t *C, int ldC, uint16_t *Apacked) {
    constexpr int VLEN = VecType<uint32_t>::size();
    for (int i = 0; i < m; i += MC) {
        int ib = std::min(MC, m - i);
        pack_K<uint8_t, uint16_t, MR, 2, true>(k, ib, &alpha(i, 0), ldA, Apacked);
        showMatrix(2, ib*k/2, Apacked, 1, "Apack");
        int kk = (k + 3) & -4;
        LoopTwo<NR>(ib, n, k, Apacked, Bpacked, &gamma(i, 0), ldC);
    }
}

constexpr int fullVectored(int N, int size) {
    return (N % (VLEN_BYTES / size)) == 0;
}

constexpr int kernelType(int ROWS, int COLS, int elementSize) {
    return fullVectored(ROWS, elementSize) ? 1 : 2;
}

template <int ROWS, int COLS, typename T, typename DT>
inline
    typename std::enable_if<kernelType(ROWS, COLS, sizeof(T)) == 1, void>::type
    gbp(int k, const DT *MP_A, const DT *MP_B, T *C, int ldC) {
    using vType = typename VecType<T>::Type;
    using elType = typename VecType<T>::ElementType;
    constexpr int VLEN = VecType<T>::size();
    const T *MT_A = reinterpret_cast<const T *>(MP_A);
    const T *MT_B = reinterpret_cast<const T *>(MP_B);

    VecType<T> Caux[ROWS / VLEN][COLS] = {};
    int real_k = k / 2;
    int real_k_4 = real_k & (-4);

    if (real_k & 3) {
        for (int p = 0; p < (real_k&3); p ++) {
        for (int i = 0; i < ROWS / VLEN; i++) {
            auto Ak = cast<T, DT>(VecType<T>::load_hinted(&MT_A[i * VLEN]));

            for (int j = 0; j < COLS; j++) {
                auto BkI = cast<T, DT>(VecType<T>{MT_B[j]});

                Caux[i][j] = multiplyAdd(Ak, BkI, Caux[i][j]);
            }
        }
        MT_A += ROWS;
        MT_B += COLS;
        }
    }

    asm("");

    for (int p = 0; p < real_k_4; p += 4) {
            for (int i = 0; i < ROWS / VLEN; i++) {
                auto Ak0 = cast<T, DT>(VecType<T>::load_hinted(&MT_A[i * VLEN]));
                auto Ak1 = cast<T, DT>(VecType<T>::load_hinted(&MT_A[i * VLEN + ROWS]));
                auto Ak2 = cast<T, DT>(VecType<T>::load_hinted(&MT_A[i * VLEN + 2*ROWS]));
                auto Ak3 = cast<T, DT>(VecType<T>::load_hinted(&MT_A[i * VLEN + 3 *ROWS]));
                for (int j = 0; j < COLS; j++) {
                    auto BkI0 = cast<T, DT>(VecType<T>{MT_B[j]});
                    auto BkI1 = cast<T, DT>(VecType<T>{MT_B[j + COLS]});
                    auto BkI2 = cast<T, DT>(VecType<T>{MT_B[j + 2 * COLS]});
                    auto BkI3 = cast<T, DT>(VecType<T>{MT_B[j + 3*COLS]});

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
#if !defined(LAST_INDEX_FAST)

    for (int j = 0; j < COLS; j++) {
        for (int i = 0; i < ROWS / VLEN; i++) {
            vType *C_ij = (vType *)&gamma(i * VLEN, j);
            *C_ij += (vType)Caux[i][j];
        }
    }
#else
#pragma GCC error "was not implemented"
#endif
}

template <int ROWS, int COLS, typename T, typename DT>
inline
    typename std::enable_if<kernelType(ROWS, COLS, sizeof(T)) == 2, void>::type
    gbp(int k, const DT *MP_A, const DT *MP_B, T *C, int ldC) {
 
    using vType = typename VecType<T>::Type;
    using elType = typename VecType<T>::ElementType;
    constexpr int VLEN = VecType<T>::size();
    const T *MT_A = reinterpret_cast<const T *>(MP_A);
    const T *MT_B = reinterpret_cast<const T *>(MP_B);

    VecType<T> Caux[COLS] = {};
    int real_k = k / 2;
    int real_k_4 = real_k & (-4);
    constexpr int BYTE_INDEX = ROWS * 2 *sizeof(DT) - 1; 
 
    if (real_k & 3) {
        for (int p = 0; p < (real_k&3); p ++) {

            auto Ak = cast<T, DT>(VecType<T>::loadLen(MT_A, BYTE_INDEX));
 
            for (int j = 0; j < COLS; j++) {
                auto BkI = cast<T, DT>(VecType<T>{MT_B[j]});

                Caux[j] = multiplyAdd(Ak, BkI, Caux[j]);
            }

        MT_A += ROWS;
        MT_B += COLS;
        }
    }

    asm("");

    for (int p = 0; p < real_k_4; p += 4) {
        
                auto Ak0 = cast<T, DT>(VecType<T>::loadLen(&MT_A[0], BYTE_INDEX));
                auto Ak1 = cast<T, DT>(VecType<T>::loadLen(&MT_A[ROWS], BYTE_INDEX));
                auto Ak2 = cast<T, DT>(VecType<T>::loadLen(&MT_A[2*ROWS], BYTE_INDEX));
                auto Ak3 = cast<T, DT>(VecType<T>::loadLen(&MT_A[3 *ROWS], BYTE_INDEX));
                for (int j = 0; j < COLS; j++) {
                    auto BkI0 = cast<T, DT>(VecType<T>{MT_B[j]});
                    auto BkI1 = cast<T, DT>(VecType<T>{MT_B[j + COLS]});
                    auto BkI2 = cast<T, DT>(VecType<T>{MT_B[j + 2 * COLS]});
                    auto BkI3 = cast<T, DT>(VecType<T>{MT_B[j + 3*COLS]});

                    Caux[j] = multiplyAdd(Ak0, BkI0, Caux[j]);
                    Caux[j] = multiplyAdd(Ak1, BkI1, Caux[j]);
                    Caux[j] = multiplyAdd(Ak2, BkI2, Caux[j]);
                    Caux[j] = multiplyAdd(Ak3, BkI3, Caux[j]);
                }
        MT_A += 4 * ROWS;
        MT_B += 4 * COLS;
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
typename std::enable_if<(M >= ROWS), void>::type LoopOne_TAIL(
    int m, int k, uint16_t *Apacked, uint16_t *Bpacked, uint32_t *C, int ldC) {
    // end of the roll
}

template <int M, int ROWS, int COLS>
typename std::enable_if<(M < ROWS), void>::type LoopOne_TAIL(
    int m, int k, uint16_t *Apacked, uint16_t *Bpacked, uint32_t *C, int ldC) {
    if (m & M) {
        gbp<M, COLS, uint32_t, uint16_t>(k, Apacked, Bpacked, C, ldC);
        Apacked = &Apacked[M * k];
        C = &gamma(M, 0);
        // std::cout<<C<<std::endl;;
    }
    LoopOne_TAIL<2 * M, ROWS, COLS>(m, k, Apacked, Bpacked, C, ldC);
}

template <int ROWS, int COLS>
inline void LoopOne(int m, int k, uint16_t *Apacked, uint16_t *Bpacked,
                    uint32_t *C, int ldC) {
    for (int i = 0; i < m / ROWS; i++) {
        gbp<ROWS, COLS, uint32_t, uint16_t>(k, &Apacked[i * ROWS * k], Bpacked,
                                           &gamma(i * ROWS, 0), ldC);
    }
    int II = m - m % ROWS;
    if (m > II)
        LoopOne_TAIL<1, ROWS, COLS>(m - II, k, &Apacked[II * k], Bpacked,
                                    &gamma(II, 0), ldC);
}

template <int N, int COLS>
typename std::enable_if<(N >= COLS), void>::type LoopTwo_TAIL(
    int m, int n, int k, uint16_t *Apacked, uint16_t *Bpacked, uint32_t *C,
    int ldC) {
    // end of the roll
}

template <int N, int COLS>
typename std::enable_if<(N < COLS), void>::type LoopTwo_TAIL(
    int m, int n, int k, uint16_t *Apacked, uint16_t *Bpacked, uint32_t *C,
    int ldC) {
    if (n & N) {
        LoopOne<MR, N>(m, k, Apacked, Bpacked, C, ldC);
        Bpacked = &Bpacked[N * k];
        C = &gamma(0, N);
    }
    LoopTwo_TAIL<2 * N, COLS>(m, n, k, Apacked, Bpacked, C, ldC);
}

template <int COLS>
inline void LoopTwo(int m, int n, int k, uint16_t *Apacked, uint16_t *Bpacked, uint32_t *C, int ldC) {
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
        LoopTwo_TAIL<1, COLS>(m, n - JJ, k, Apacked, &Bpacked[JJ * k],
                              &gamma(0, JJ), ldC);
}
