#pragma once
#include <iostream>
#include <utils.h>

template <const int MR, typename T>
void pack_k4_MR(int m, int k, T *A, int ldA, T *Adest)
{
    if (m == MR)
    {
        int k_4 = k & -4;
        for (int p = 0; p < k_4; p += 4)
        {
            for (int i = 0; i < MR; i++)
            {
                Adest[0] = alpha(i, p);
                Adest[1] = alpha(i, p + 1);
                Adest[2] = alpha(i, p + 2);
                Adest[3] = alpha(i, p + 3);
                Adest += 4;
            }
        }
        if ((k & 3) == 3)
        {
            for (int i = 0; i < MR; i++)
            {
                Adest[0] = alpha(i, k_4);
                Adest[1] = alpha(i, k_4 + 1);
                Adest[2] = alpha(i, k_4 + 2);

                Adest[3] = 0;
                Adest += 4;
            }
        }
        else if ((k & 2) == 2)
        {
            for (int i = 0; i < MR; i++)
            {
                Adest[0] = alpha(i, k_4);
                Adest[1] = alpha(i, k_4 + 1);
                Adest[2] = 0;
                Adest[3] = 0;
                Adest += 4;
            }
        }
        else if ((k & 1) == 1)
        {
            for (int i = 0; i < MR; i++)
            {
                Adest[0] = alpha(i, k_4);
                Adest[1] = 0;
                Adest[2] = 0;
                Adest[3] = 0;
                Adest += 4;
            }
        }
    }
    else
    {

        int k_4 = k & -4;
        
        for (int p = 0; p < k_4; p += 4)
        {
            for (int i = 0; i < m; i++)
            {
                
                Adest[0] = alpha(i, p);
                Adest[1] = alpha(i, p + 1);
                Adest[2] = alpha(i, p + 2);
                Adest[3] = alpha(i, p + 3);
                Adest += 4;
            }
            for (int i = m; i < MR; i++)
            {
                
                Adest[0] = 0;
                Adest[1] = 0;
                Adest[2] = 0;
                Adest[3] = 0;
                Adest += 4;
            }
        }
        if ((k & 3) == 3)
        {
            for (int i = 0; i < m; i++)
            {
                Adest[0] = alpha(i, k_4);
                Adest[1] = alpha(i, k_4 + 1);
                Adest[2] = alpha(i, k_4 + 2);
                Adest[3] = 0; 
                Adest += 4;
            }
            for (int i = m; i < MR; i++)
            {
                Adest[0] = 0;
                Adest[1] = 0;
                Adest[2] = 0;
                Adest[3] = 0;
                Adest += 4;
            }
        }
        else if ((k & 2) == 2)
        {
            for (int i = 0; i < m; i++)
            {
                Adest[0] = alpha(i, k_4);
                Adest[1] = alpha(i, k_4 + 1);
                Adest[2] = 0;
                Adest[3] = 0;
                Adest += 4;
            }
            for (int i = m; i < MR; i++)
            {
                Adest[0] = 0;
                Adest[1] = 0;
                Adest[2] = 0;
                Adest[3] = 0;
                Adest += 4;
            }
        }
        else if ((k & 1) == 1)
        {
            for (int i = 0; i < m; i++)
            {
                Adest[0] = alpha(i, k_4);
                Adest[1] = 0;
                Adest[2] = 0;
                Adest[3] = 0;
                Adest += 4;
            }
            for (int i = m; i < MR; i++)
            {
                Adest[0] = 0;
                Adest[1] = 0;
                Adest[2] = 0;
                Adest[3] = 0;
                Adest += 4;
            }
        }
    }
}

template <typename T, int MR>
void PackBlockA_MCxKC(int m, int k, T *A, int ldA, T *Adest)
{
    // if k is not divisible by 4 , the rest will be 0-ed
    int kk = (k + 3) & (-4);
    for (int i = 0; i < m; i += MR)
    {
        int ib = std::min(MR, m - i);
        
        pack_k4_MR<MR, T>(ib, k, &alpha(i, 0), ldA, Adest);
        Adest += ib * kk;
    }
}



template <const int NR, typename T>
void pack_k4_NR(int k, int n, T *B, int ldB, T *Bdest)
{

    if (n == NR)
    {
     //       std::cout<<"v bn "<<n<<std::endl;
        int k_4 = k & -4;
        for (int p = 0; p < k_4; p += 4)
        {
            for (int j = 0; j < NR; j++)
            {
                Bdest[0] = beta( p, j);
                Bdest[1] = beta( p + 1, j);
                Bdest[2] = beta( p + 2, j);
                Bdest[3] = beta( p + 3, j);
                Bdest += 4;
            }
        }
        if ((k & 3) == 3)
        {
            for (int j = 0; j < NR; j++)
            {
                Bdest[0] = beta( k_4, j);
                Bdest[1] = beta( k_4 + 1, j);
                Bdest[2] = beta( k_4 + 2, j);
                Bdest[3] = 0;
                Bdest += 4;
            }
        }
        else if ((k & 2) == 2)
        {
            for (int j = 0; j < NR; j++)
            {
                Bdest[0] = beta( k_4, j);
                Bdest[1] = beta( k_4 + 1, j);
                Bdest[2] = 0;
                Bdest[3] = 0;
                Bdest += 4;
            }
        }
        else if ((k & 1) == 1)
        {
            for (int j = 0; j < NR; j++)
            {
                Bdest[0] = beta( k_4, j);
                Bdest[1] = 0;
                Bdest[2] = 0;
                Bdest[3] = 0;
                Bdest += 4;
            }
        }
    }
    else
    {

        int k_4 = k & -4;
    // std::cout<<"else bn "<<n<< " k_4 "<<k_4<<std::endl;
        for (int p = 0; p < k_4; p += 4)
        {
            for (int j = 0; j < n; j++)
            {
 
                Bdest[0] = beta( p, j);
                Bdest[1] = beta( p + 1, j);
                Bdest[2] = beta( p + 2, j);
                Bdest[3] = beta( p + 3, j);
                Bdest += 4;
            }
            for (int j = n; j < NR; j++)
            {
                
                Bdest[0] = 0;
                Bdest[1] = 0;
                Bdest[2] = 0;
                Bdest[3] = 0;
                Bdest += 4;
            }
        }
        if ((k & 3) == 3)
        {
              //  std::cout<<"(k & 3) == 3 "<<k<<" ; "<<n <<std::endl;
            for (int j = 0; j < n; j++)
            {
                Bdest[0] = beta( k_4, j);
                Bdest[1] = beta( k_4 + 1, j);
                Bdest[2] = beta( k_4 + 2, j);
                Bdest[3] = 0;
//std::cout<<Bdest[0] <<","<<Bdest[1]<<","<<Bdest[2]<<","<<Bdest[3]<<std::endl;
                Bdest += 4;
            }
            for (int j = n; j < NR; j++)
            {
                Bdest[0] = 0;
                Bdest[1] = 0;
                Bdest[2] = 0;
                Bdest[3] = 0;
                Bdest += 4;
                //                std::cout<<Bdest[0] <<","<<Bdest[1]<<","<<Bdest[2]<<","<<Bdest[3]<<std::endl;
            }
        }
        else if ((k & 2) == 2)
        {
           // std::cout<<"(k & 2) == 2"<<k <<std::endl;
            for (int j = 0; j < n; j++)
            {
                Bdest[0] = beta( k_4, j);
                Bdest[1] = beta( k_4 + 1, j);
                Bdest[2] = 0;
                Bdest[3] = 0;
                Bdest += 4;
            }
            for (int j = n; j < NR; j++)
            {
                Bdest[0] = 0;
                Bdest[1] = 0;
                Bdest[2] = 0;
                Bdest[3] = 0;
                Bdest += 4;
            }
        }
        else if ((k & 1) == 1)
        {
             // std::cout<<"(k & 1) == 1"<<k <<std::endl;
            for (int j = 0; j < n; j++)
            {
                Bdest[0] = beta( k_4, j);
                Bdest[1] = 0;
                Bdest[2] = 0;
                Bdest[3] = 0;
                Bdest += 4;
            }
            for (int j = n; j < NR; j++)
            {
                Bdest[0] = 0;
                Bdest[1] = 0;
                Bdest[2] = 0;
                Bdest[3] = 0;
                Bdest += 4;
            }
        }
    }
}

template <typename T, int NR>
void PackBlockB_KCxNC(int k, int n, T *B, int ldB, T *Bdest)
{
    // if k is not divisible by 4 , the rest will be 0-ed
    int kk = (k + 3) & (-4);
    for (int j = 0; j < n; j += NR)
    {
        int jb = std::min(NR, n - j) ;

        pack_k4_NR<NR, T>(k, jb, &beta( 0, j), ldB, Bdest);
        Bdest += jb * kk;
    }
}

