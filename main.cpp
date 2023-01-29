
#include <utils.h>

#include <iostream>
#include <memory>
//#include <pack.hpp>
#include <test_reference.h>

void gemmu8u8s32(const char *transa, const char *transb,
        const char *offsetc,dim_t M,dim_t N,dim_t K,
       float alpha, const uint8_t *A,dim_t LDA, const uint8_t *ao,
        const uint8_t *B,dim_t LDB, const uint8_t *bo,
       float beta, int32_t *C, dim_t LDC, const int32_t *co) ;

extern double FLA_Clock();
constexpr int32_t C_VAL = 12;
double validate(const char* transA, const char * transB, const char *offsetc,  int m, int n, int k,
 float alpha, uint8_t *A, int ldA,const uint8_t *ao,
 uint8_t *B, int ldB,  const uint8_t *bo,
 float beta,  int32_t *C, int ldC,  const int32_t *co) {
    test_params p = {};
    p.transA = *transA;
    p.transB = *transB;
    p.M = m;
    p.K = k;
    p.N = n;
    p.alpha = alpha;
    p.beta = beta;
    p.lda = ldA;
    p.ldb = ldB;
    ;
    p.igemm_params = {};
    p.off = {};
    int32_t oc[1] = {};
    int ldCref = determineLd(m, n, 0);
    p.ldc = ldCref;
    std::unique_ptr<int32_t[]> Cref(new int32_t[determineSize(m, n, 0)]);
    fillMatrix(m, n, Cref.get(), ldC, C_VAL);
    ref_gemm<uint8_t, uint8_t>::call(p, m, n, A, B, Cref.get(), oc);
    showMatrix(m, n, Cref.get(), m, "Cref");

    return maxAbsDiff<double>(m, n, Cref.get(), ldCref, C, ldC);
}
 
int main() {
    // constexpr int ldPlus = 1032;
    //   //for(int n =1000; n<1033;n++){
    //   constexpr int m=1000;
    //   constexpr int n=1000;
    //   constexpr int k=1000;
    dim_t last =   1200;
    dim_t first =    200;
    dim_t inc = 200;
    dim_t nrepeats = 4;
    TestSeed seed{};
    dim_t m, n, k;
    char trans[]={'n','t'};
    float alpha = 1.0;
    float beta =  0.0;
    const char *offsetC =" ";
    const int32_t *co = nullptr;
    const uint8_t *ao = nullptr;
    const uint8_t *bo = nullptr;
    for(auto transA : trans)
       for(auto transB : trans) {
    std::cout << "seed " << seed << " MR: " << MR << " NR: " << NR << " -- transA: "<<transA<<" transB: "<<transB<<std::endl;
    printf("%%          time       G_OPS     diff \n");
    for (dim_t size = last; size >= first; size -= inc) {
        /* we will only time cases where all three matrices are square */
        m = n = k = size;
        // m=1;
        // n=1;
        dim_t ldA = determineLd(m, k, size, transA);
        dim_t ldB = determineLd(k, n, size, transB);
        dim_t ldC = determineLd(m, n, size);
        // std::cout<<m<<","<<n<<","<<ldC<<","<<determineSize(m,n,ldC)<<std::endl;

        std::unique_ptr<uint8_t[]> A(new uint8_t[determineSize(m, k, ldA,transA)]);
        std::unique_ptr<uint8_t[]> B(new uint8_t[determineSize(k, n, ldB, transB)]);
        std::unique_ptr<int32_t[]> C(new int32_t[determineSize(m, n, ldC)]);

#if !defined(LOW_0_127)
        randomMatrix(m, k, A.get(), ldA, seed);
#else
        randomMatrix<uint8_t>(m, k, A.get(), ldA, 0, 127, seed);
#endif
        randomMatrix(k, n, B.get(), ldB, seed);
        //    linMatrix(m,k, A.get(),ldA, (uint8_t)1);
        //      linMatrix(k,n, B.get(),ldB, (uint8_t)100);

        showMatrix(m, k, A.get(), ldA, "A");
        showMatrix(k, n, B.get(), ldB, "B");

        double dtime, dtime_best;
        //   showMatrix( ((m+MR-1) & (-MR)),((k+3) & (-4)) ,Apack.get(),
        //   ((m+MR-1) & (-MR)), "Apack");

        //   showMatrix(((k+3) & (-4)), ((n+NR-1) & (-NR)) ,Bpack.get(), ((k+3)
        //   & (-4)), "Bpack");

        //    gbp<MR, NR, int32_t, uint8_t>(((k+3) & (-4)), Apack.get(),
        //    Bpack.get(), C.get(), ldC);
        auto gops = 2.0 * m * n * k * 1e-09;
        // std::cout<<"//Begin//"<<std::endl;
        int irep = 0;
        for (irep = 0; irep < nrepeats; irep++) {
            fillMatrix(m, n, C.get(), ldC, C_VAL);
            auto dtime = FLA_Clock();
            gemmu8u8s32(&transA,&transB, offsetC, m, n, k, alpha, A.get(), ldA, ao, B.get(), ldB, bo, beta, C.get(), ldC, co);
            dtime = FLA_Clock() - dtime;
            if (irep == 0)
                dtime_best = dtime;
            else
                dtime_best = (dtime < dtime_best ? dtime : dtime_best);
        }
        
        auto diff = validate(&transA,&transB, offsetC, m, n, k, alpha, A.get(), ldA, ao, B.get(), ldB, bo, beta, C.get(), ldC, co);
        showMatrix(m, n, C.get(), ldC, "C");
        printf("%5d %8.4le %8.4le %8.4le  \n", n, dtime_best, gops / dtime_best,
               diff);
        // std::cout<<"//End"<<std::endl;
    }

       }
    return 0;
}
