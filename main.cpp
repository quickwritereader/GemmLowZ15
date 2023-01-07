
#include <utils.h>

#include <iostream>
#include <memory>
//#include <pack.hpp>
#include <test_reference.h>
void MyGemm(int m, int n, int k, uint8_t *A, int ldA, uint8_t *B, int ldB,
            uint32_t *C, int ldC);

extern double FLA_Clock();

double validate(int m, int n, int k, uint8_t *A, int ldA, uint8_t *B, int ldB,
                int32_t *C, int ldC) {
    test_params p = {};
    p.transA = 'n';
    p.transB = 'n';
    p.M = m;
    p.K = k;
    p.N = n;
    p.alpha = 1.0;
    p.beta = 0.0;
    p.lda = ldA;
    p.ldb = ldB;
    ;
    p.igemm_params = {};
    p.off = {};
    int32_t oc[1] = {};
    int ldCref = determineLd(m, n, 0);
    p.ldc = ldCref;
    std::unique_ptr<int32_t[]> Cref(new int32_t[determineSize(m, n, 0)]);

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
    int last =  1200;
    int first =   200;
    int inc = 200;
    int nrepeats =  4;
    TestSeed seed{};
    int m, n, k;
    std::cout << "seed " << seed << " MR: " << MR << " NR: " << NR << std::endl;
    printf("%%          time       G_OPS     diff \n");
    for (int size = last; size >= first; size -= inc) {
        /* we will only time cases where all three matrices are square */
        m = n = k = size;
        // m=1;
        // n=1;
        int ldA = determineLd(m, k, size);
        int ldB = determineLd(k, n, size);
        int ldC = determineLd(m, n, size);
        // std::cout<<m<<","<<n<<","<<ldC<<","<<determineSize(m,n,ldC)<<std::endl;

        std::unique_ptr<uint8_t[]> A(new uint8_t[determineSize(m, k, ldA)]);
        std::unique_ptr<uint8_t[]> B(new uint8_t[determineSize(k, n, ldB)]);
        std::unique_ptr<uint32_t[]> C(new uint32_t[determineSize(m, n, ldC)]);

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

        //    gbp<MR, NR, uint32_t, uint8_t>(((k+3) & (-4)), Apack.get(),
        //    Bpack.get(), C.get(), ldC);
        auto gops = 2.0 * m * n * k * 1e-09;
        // std::cout<<"//Begin//"<<std::endl;
        int irep = 0;
        for (irep = 0; irep < nrepeats; irep++) {
            fillMatrix(m, n, C.get(), ldC, (uint32_t)0);
            auto dtime = FLA_Clock();
            MyGemm(m, n, k, A.get(), ldA, B.get(), ldB, C.get(), ldC);
            dtime = FLA_Clock() - dtime;
            if (irep == 0)
                dtime_best = dtime;
            else
                dtime_best = (dtime < dtime_best ? dtime : dtime_best);
        }
        auto diff = validate(m, n, k, A.get(), ldA, B.get(), ldB,
                             (int32_t *)C.get(), ldC);
        showMatrix(m, n, C.get(), ldC, "C");
        printf("%5d %8.4le %8.4le %8.4le  \n", n, dtime_best, gops / dtime_best,
               diff);
        // std::cout<<"//End"<<std::endl;
    }
    return 0;
}