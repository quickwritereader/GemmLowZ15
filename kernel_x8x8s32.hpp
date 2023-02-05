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
    return fullVectored(ROWS, elementSize) ? (COLS%4==0?0:1) : 2;
}



template <int ROWS, int COLS>
inline typename std::enable_if<kernelType(ROWS, COLS, sizeof(uint32_t)) == 0,
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
                    Caux[i][j] += multiplySum4(Ak, BkI);
#else                    
                    Caux[i][j] += multiplySum4Low(Ak, BkI);
#endif 
                }
            }
            MT_A += ROWS;
            MT_B += COLS;
    }

    asm("");
    dim_t real_k_2 = real_k & (-2);
     const vuint16 vz1 = {1,1,1,1,1,1,1,1};
    for (dim_t p = 0; p < real_k_2; p += 2) {

       
#if !defined(LOW_0_127)
     static_assert((COLS%4==0),"COL should be divisible by 4");
            for (int j = 0; j < COLS/4; j+=4) {
                auto Bk0_0 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j]});
                auto Bk0_1 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j+1]});
                auto Bk0_2 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j+2]});
                auto Bk0_3 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j+3]});
                auto Bk1_0 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j+COLS]});
                auto Bk1_1 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j+COLS+1]});
                auto Bk1_2 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j+COLS+2]});
                auto Bk1_3 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j+COLS+3]});
           for (int i = 0; i < ROWS / VLEN; i++) {
            auto Ak0 = cast<uint8_t>(vec_type_t<uint32_t>::load_hinted(&MT_A[i * VLEN]));                 
                const vuint8 a0 = Ak0; 
                
                const vuint8 b0_0 = Bk0_0;
                const vuint8 b0_1 = Bk0_1;
                const vuint8 b0_2 = Bk0_2;
                const vuint8 b0_3 = Bk0_3;    
                vuint16 reso0_0 = vec_mulo(a0, b0_0);
                vuint16 reso0_1 = vec_mulo(a0, b0_1);
                vuint16 reso0_2 = vec_mulo(a0, b0_2);
                vuint16 reso0_3 = vec_mulo(a0, b0_3);
         

                vuint16 rese0_0 = vec_mule(a0, b0_0);
                vuint16 rese0_1 = vec_mule(a0, b0_1);
                vuint16 rese0_2 = vec_mule(a0, b0_2);
                vuint16 rese0_3 = vec_mule(a0, b0_3);
   
                

                Caux[i][j] = vec_moadd(reso0_0, vz1, Caux[i][j].vec());
                Caux[i][j+1] = vec_moadd(reso0_1, vz1, Caux[i][j+1].vec());
                Caux[i][j+2] = vec_moadd(reso0_2, vz1, Caux[i][j+2].vec());
                Caux[i][j+3] = vec_moadd(reso0_3, vz1, Caux[i][j+3].vec());
     
                Caux[i][j] = vec_meadd(reso0_0, vz1, Caux[i][j].vec());
                Caux[i][j+1] = vec_meadd(reso0_1, vz1, Caux[i][j+1].vec());
                Caux[i][j+2] = vec_meadd(reso0_2, vz1, Caux[i][j+2].vec());
                Caux[i][j+3] = vec_meadd(reso0_3, vz1, Caux[i][j+3].vec());
                    
               

        
           

                Caux[i][j] = vec_moadd(rese0_0, vz1, Caux[i][j].vec());
                Caux[i][j+1] = vec_moadd(rese0_1, vz1, Caux[i][j+1].vec());
                Caux[i][j+2] = vec_moadd(rese0_2, vz1, Caux[i][j+2].vec());
                Caux[i][j+3] = vec_moadd(rese0_3, vz1, Caux[i][j+3].vec());
 
                Caux[i][j] = vec_meadd(rese0_0, vz1, Caux[i][j].vec());
                Caux[i][j+1] = vec_meadd(rese0_1, vz1, Caux[i][j+1].vec());
                Caux[i][j+2] = vec_meadd(rese0_2, vz1, Caux[i][j+2].vec());
                Caux[i][j+3] = vec_meadd(rese0_3, vz1, Caux[i][j+3].vec());
   auto Ak1 = cast<uint8_t>(vec_type_t<uint32_t>::load_hinted(&MT_A[i * VLEN + ROWS]));                 
                     const vuint8 a1 = Ak1; 
                
                const vuint8 b1_0 = Bk1_0;
                const vuint8 b1_1 = Bk1_1;
                const vuint8 b1_2 = Bk1_2;
                const vuint8 b1_3 = Bk1_3;    
                vuint16 reso1_0 = vec_mulo(a1, b1_0);
                vuint16 reso1_1 = vec_mulo(a1, b1_1);
                vuint16 reso1_2 = vec_mulo(a1, b1_2);
                vuint16 reso1_3 = vec_mulo(a1, b1_3);
         
                vuint16 rese1_0 = vec_mule(a1, b1_0);
                vuint16 rese1_1 = vec_mule(a1, b1_1);
                vuint16 rese1_2 = vec_mule(a1, b1_2);
                vuint16 rese1_3 = vec_mule(a1, b1_3);
 
                

                Caux[i][j] = vec_moadd(reso1_0, vz1, Caux[i][j].vec());
                Caux[i][j+1] = vec_moadd(reso1_1, vz1, Caux[i][j+1].vec());
                Caux[i][j+2] = vec_moadd(reso1_2, vz1, Caux[i][j+2].vec());
                Caux[i][j+3] = vec_moadd(reso1_3, vz1, Caux[i][j+3].vec());
 
                Caux[i][j] = vec_meadd(reso1_0, vz1, Caux[i][j].vec());
                Caux[i][j+1] = vec_meadd(reso1_1, vz1, Caux[i][j+1].vec());
                Caux[i][j+2] = vec_meadd(reso1_2, vz1, Caux[i][j+2].vec());
                Caux[i][j+3] = vec_meadd(reso1_3, vz1, Caux[i][j+3].vec());
                    
 


        
 
                Caux[i][j] = vec_moadd(rese1_0, vz1, Caux[i][j].vec());
                Caux[i][j+1] = vec_moadd(rese1_1, vz1, Caux[i][j+1].vec());
                Caux[i][j+2] = vec_moadd(rese1_2, vz1, Caux[i][j+2].vec());
                Caux[i][j+3] = vec_moadd(rese1_3, vz1, Caux[i][j+3].vec());
 
                Caux[i][j] = vec_meadd(rese1_0, vz1, Caux[i][j].vec());
                Caux[i][j+1] = vec_meadd(rese1_1, vz1, Caux[i][j+1].vec());
                Caux[i][j+2] = vec_meadd(rese1_2, vz1, Caux[i][j+2].vec());
                Caux[i][j+3] = vec_meadd(rese1_3, vz1, Caux[i][j+3].vec());
                 
           }
            }  

#else 
        for (int i = 0; i < ROWS / VLEN; i++) {
            auto Ak0 = cast<uint8_t>(vec_type_t<uint32_t>::load_hinted(&MT_A[i * VLEN]));
            auto Ak1 = cast<uint8_t>(
                    vec_type_t<uint32_t>::load_hinted(&MT_A[i * VLEN + ROWS])); 
                    //assume COL2 is divisible
        for (int j = 0; j < COLS; j++) {
            auto BkI0 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j]});
            auto BkI1 = cast<uint8_t>(vec_type_t<uint32_t> {MT_B[j + COLS]});
            Caux[i][j] += multiplySum4Low(Ak0, BkI0);
            Caux[i][j] += multiplySum4Low(Ak1, BkI1);
                        }
        }
#endif


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
                    Caux[i][j] += multiplySum4(Ak, BkI);
#else                    
                    Caux[i][j] += multiplySum4Low(Ak, BkI);
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

                Caux[i][j] += vec_sum4(reso1, reso0);
                Caux[i][j] += vec_sum4(rese1, rese0);
                Caux[i][j] += vec_sum4(resh, vz16);
#else 
            Caux[i][j] += multiplySum4Low(Ak0, BkI0);
            Caux[i][j] += multiplySum4Low(Ak1, BkI1);
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
                Caux[j] += multiplySum4(Ak, BkI);
#else
                Caux[j] += multiplySum4Low(Ak, BkI);
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

                Caux[j] += vec_sum4(reso1, reso0);
                Caux[j] += vec_sum4(rese1, rese0);
                Caux[j] += vec_sum4(resh, vz16); 
#else
            Caux[j] += multiplySum4Low(Ak0, BkI0);
            Caux[j] += multiplySum4Low(Ak1, BkI1);

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
