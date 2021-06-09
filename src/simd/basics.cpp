/* 

In this tutorial we will use a cpu capable of handling 256 bit registers. 
The maximum we can hold in each register is as follows
->  4 x 64bit float/int
->  8 x 32bit float/int
-> 16 x 16bit int
-> 32 x  8bit int/uint

Helper functions:

cast
mask/perm map
load 
store
blend
perm
min max
+ - / * 
mask

*/


#include <immintrin.h>
#include <iostream>
#include <assert.h>
#include "../Timer.h"

using uchar = unsigned char;
using namespace std;


__m256i castps_32i(__m256 a)
{
    return _mm256_cvtps_epi32(a);
}

__m256 cast32i_ps(__m256i a)
{
    return _mm256_cvtepi32_ps(a);
}

__m256 load_from(float const* ptr);

template <unsigned A = 0, unsigned B = 0, unsigned C = 0, unsigned D = 0,
          unsigned E = 0, unsigned F = 0, unsigned G = 0, unsigned H = 0>
__m256i make_mask_map()
{
    static_assert(A < 2 && B < 2 && C < 2 && D < 2 &&
                  E < 2 && F < 2 && G < 2 && H < 2);
    
    return _mm256_set_epi32(A, B, C, D, E, F, G, H);
}

template <unsigned A = 0, unsigned B = 0, unsigned C = 0, unsigned D = 0,
          unsigned E = 0, unsigned F = 0, unsigned G = 0, unsigned H = 0>
__m256i make_perm_map()
{
    static_assert(A < 8 && B < 8 && C < 8 && D < 8 &&
                  E < 8 && F < 8 && G < 8 && H < 8);
    
    return _mm256_set_epi32(A, B, C, D, E, F, G, H);
}

__m256i make_perm_map_rt(short A = 0, short B = 0, short C = 0, short D = 0,
          short E = 0, short F = 0, short G = 0, short H = 0)
          {
                assert(A < 8 && B < 8 && C < 8 && D < 8 &&
                            E < 8 && F < 8 && G < 8 && H < 8);

                float a[8];
                a[0] = A;
                a[1] = B;
                a[2] = C;
                a[3] = D;
                a[4] = E;
                a[5] = F;
                a[6] = G;
                a[7] = H;
                return castps_32i(load_from(a));
          }

__m256i make_mask_map_rt(unsigned A = 0, unsigned B = 0, unsigned C = 0, unsigned D = 0,
          unsigned E = 0, unsigned F = 0, unsigned G = 0, unsigned H = 0)
          {
                assert(A < 2 && B < 2 && C < 2 && D < 2 &&
                            E < 2 && F < 2 && G < 2 && H < 2);
                // return _mm256_set_epi32(A, B, C, D, E, F, G, H);
                float a[8];
                a[0] = A;
                a[1] = B;
                a[2] = C;
                a[3] = D;
                a[4] = E;
                a[5] = F;
                a[6] = G;
                a[7] = H;
                return castps_32i(load_from(a));
          }

__m256 load_value(float val)
{
    return _mm256_set1_ps(val);
}

__m256 load_from(float const* ptr)
{
    return _mm256_loadu_ps(ptr);
}

inline __m256 masked_load_from(float const* ptr, __m256i mask)
{
    return _mm256_maskload_ps(ptr, mask);
}

void store_to(float* ptr, __m256 reg)
{
    _mm256_storeu_ps(ptr, reg);
}


inline void masked_store_to(float* ptr, __m256 reg, __m256i mask)
{
    _mm256_maskstore_ps(ptr, mask, reg);
}

inline __m256 blend(__m256 a, __m256 b, __m256i mask)
{
    return _mm256_blendv_ps(a, b, cast32i_ps(mask));
}

inline __m256 permute(__m256 a, __m256i idx)
{
    return _mm256_permutevar8x32_ps(a, idx);
}

inline __m256 mask_permute(__m256 a, __m256i perm, __m256i mask)
{
    __m256 ap = permute(a, perm);
    return blend(a, ap, mask);
}

__m256 rotate(__m256 a, int R) // clockwise positive S
{
    if(R % 8 == 0)
        return a;
    
    short S = R % 8;
    short A = (S + 0) % 8;
    short B = (S + 1) % 8;
    short C = (S + 2) % 8;
    short D = (S + 3) % 8;
    short E = (S + 4) % 8;
    short F = (S + 5) % 8;
    short G = (S + 6) % 8;
    short H = (S + 7) % 8;

    return permute(a, make_perm_map_rt(A, B, C, D, E, F, G, H));
}

__m256 shift(__m256 a, int S);

__m256 shift_carry_up(__m256 a, __m256 b, int S)
{
    if(S > 8)
        return shift(a, S % 8);
    else
    {
        // b = rotate(b, -S);
        // a = rotate(a, -S);
        short A = (S > 0) ? 1 : 0;
        short B = (S > 1) ? 1 : 0;
        short C = (S > 2) ? 1 : 0;
        short D = (S > 3) ? 1 : 0;
        short E = (S > 4) ? 1 : 0;
        short F = (S > 5) ? 1 : 0;
        short G = (S > 6) ? 1 : 0;
        short H = (S > 7) ? 1 : 0;
        // __m256i mask = make_mask_map_rt(A, B, C, D, E, F, G, H);
        // return blend(b, a, mask);
        return blend(b, a, _mm256_set_epi32(A, B, C, D, E, F, G, H));
    }
}

__m256 shift_carry_down(__m256 a, __m256 b, int S)
{
    if(S > 8)
        return shift(a, S % 8);
    else
    {
        a = rotate(a, S);
        b = rotate(b, S);
        short A = (S > 0) ? 1 : 0;
        short B = (S > 1) ? 1 : 0;
        short C = (S > 2) ? 1 : 0;
        short D = (S > 3) ? 1 : 0;
        short E = (S > 4) ? 1 : 0;
        short H = (S > 7) ? 1 : 0;
        short F = (S > 5) ? 1 : 0;
        short G = (S > 6) ? 1 : 0;
        // __m256i mask = make_mask_map_rt(A, B, C, D, E, F, G, H);
        // return blend(a, b, mask);
    }
}


__m256 shift(__m256 a, int S)
{
    if(S > 0)
        return shift_carry_up(a, load_value(0), S);
    return shift_carry_down(a, load_value(0), -S);
}

inline __m256 min_reg(__m256 a, __m256 b)
{
    return _mm256_min_ps(a, b);
}

inline __m256 max_reg(__m256 a, __m256 b)
{
    return _mm256_max_ps(a, b);
}

__m256 compare_and_exchange(__m256 a, __m256i perm, __m256i mask)
{
    __m256 exch = permute(a, perm);
    __m256 vmin = min_reg(a, exch);
    __m256 vmax = max_reg(a, exch);
    
    return blend(vmin, vmax, mask);
}

void sort_two_lanes_half(__m256 a)
{

}

void print(__m256 a)
{
    float* ptr;
    store_to(ptr, a);
    cout << "Register: ";
    for (int i = 0; i < 8; i++)
        cout << *(ptr + i) << " ";
    cout << endl;
    
}

void print(__m256i a)
{
    int* ptr;
    cout << "Register: ";
    store_to((float*)ptr, cast32i_ps(a));
    for (int i = 0; i < 8; i++)
        cout << *(ptr + i) << " ";
    cout << endl;
    
}


int main()
{
    int tt = 1;
    for (int pp = 0; pp < tt; pp++)
    {
        int n = 8;

        float a[n];
        for (int i = 0; i < n; i++)
            cin >> a[i];

        __m256 reg = load_from(a);
        shift(reg, 3);
        print(reg);
        // __m256i mask = make_mask_map_rt(0, 1, 1, 0, 0, 0, 1, 0);


        
        
        
    }
    

    return 0;
}
