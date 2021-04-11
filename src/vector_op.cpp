#include <iostream>
#include <immintrin.h>
#include <time.h>

#include "Timer.h"

using namespace std;


const int TIME = 1000;


struct Vector
{
    long long _len;
    float *_vector;

    Vector()
    {}
    
    Vector(long long n)
        : _len(n)
    {
        _vector = new float[n];
    }

    Vector(long long n, float* p)
        : _len(n)
    {
        _vector = new float[n];
        for (long long i = 0; i < n; i++)
            _vector[i] = p[i];

    }
    
    void fill_rand()
    {
        srand(time(0));
        for (long long i = 0; i < _len; i++)
            _vector[i] = rand()%100;
        
    }



    void add(const Vector& b, Vector& dst)
    {
        Timer time(TIME);
        dst._len = _len;
        dst._vector = new float[_len];
        for (long long i = 0; i < _len; i++)
        {
            dst._vector[i] = _vector[i] + b._vector[i];
        }
        cout << "Using normal add function: ";
    }

    void addsimd(const Vector& b, Vector& dst)
    {
        Timer time(TIME);
        dst._len = _len;
        dst._vector = new float[_len];

        /*
            In this function, we will use instruction set AVX, which supports 256 bit 
            register. Values in this register can be operated upon simulataneously.

            The Vector structure contains float data in our implementation. 
            Also, each float is 8 * 4 = 32 bit long. Thus a 256 register can pack
            256 / 32 = 8 float at once.

            If our vector is n float long, we will have to loop through iter = n / 8 times;
            If n is not a multiple of 8, we will have to compute the "tail" separately, although 
            directly doing it works the same. There might however be an invalid pointer and 
            that may cause segmentation error.
        */

        long long iter = _len / 8;
        float* pa = _vector;
        float* pb = b._vector;  

        for (long long i = 0; i < iter; i++)
        {
            //loading 8 32bit data, starting from pa + i*8 tp pa + (i+1)*8 
            __m256 a = _mm256_loadu_ps((pa + i*8)); 
            __m256 b = _mm256_loadu_ps((pb + i*8));
            
            //storing the result in dst
            _mm256_storeu_ps(dst._vector + i*8, _mm256_add_ps(a,b));
        }

        for (long long i = iter*8; i < _len; i++)
        {
            dst._vector[i] = *(pa + i) + *(pb + i);
        }
        

        cout << "Using simd add funtion: ";
    }

    void mul(const Vector& b, Vector& dst)
    {
        Timer time(TIME);
        dst._len = _len;
        dst._vector = new float[_len];
        for (long long i = 0; i < _len; i++)
        {
            dst._vector[i] = _vector[i] * b._vector[i];
        }
        cout << "Using normal mul funtion: ";
    }

    void mulsimd(const Vector& b, Vector& dst)
    {
        Timer time(TIME);
        dst._len = _len;
        dst._vector = new float[_len];

        /*
            In this function, we will use instruction set AVX, which supports 256 bit 
            register. Values in this register can be operated upon simulataneously.

            The Vector structure contains float data in our implementation. 
            Also, each float is 8 * 4 = 32 bit long. Thus a 256 register can pack
            256 / 32 = 8 float at once.

            If our vector is n float long, we will have to loop through iter = n / 8 times;
            If n is not a multiple of 8, we will have to compute the "tail" separately, although 
            directly doing it works the same. There might however be an invalid pointer and 
            that may cause segmentation error.
        */

        long long iter = _len / 8;
        float* pa = _vector;
        float* pb = b._vector;  

        for (long long i = 0; i < iter; i++)
        {
            //loading 8 32bit data, starting from pa + i*8 tp pa + (i+1)*8 
            __m256 a = _mm256_loadu_ps((pa + i*8)); 
            __m256 b = _mm256_loadu_ps((pb + i*8));

            //storing the result in dst
            _mm256_storeu_ps(dst._vector + i*8, _mm256_mul_ps(a,b));
        }

        for (long long i = iter*8; i < _len; i++)
        {
            dst._vector[i] = (*(pa + i)) * (*(pb + i));
        }
        

        cout << "Using simd mul funtion: ";
    }

    static void compare(const Vector& a, const Vector& b)
    {
        for(int i = 0; i < a._len; i++)
        {
            if(a._vector[i] != b._vector[i])
            {
                cout << "Vectors not equal\n";
                return;
            }
        }
        cout << "Vectors are equal\n";
    }
    
    ~Vector()
    {
        delete [] _vector;
    }
};


int main()
{
    Vector a(1e8 + 3);
    Vector b(1e8 + 3);

    a.fill_rand();
    b.fill_rand();
    
    Vector dst1;
    a.add(b, dst1);

    Vector dst2;
    a.addsimd(b, dst2);

    Vector::compare(dst1, dst2);

    a.mul(b, dst1);
    a.mulsimd(b, dst2);

    Vector::compare(dst1, dst2);

}