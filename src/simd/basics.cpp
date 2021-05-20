#include <immintrin.h>
#include <chrono>
#include <iostream>

struct Timer{

    std::chrono::_V2::system_clock::time_point start;
    static int cnt;
    Timer()
    {
        start = std::chrono::high_resolution_clock::now();
        cnt++;
    }

    ~Timer()
    {
        cnt--;
        float duration = std::chrono::duration<float >(std::chrono::high_resolution_clock::now() - start);
        if(cnt == 0)
        {
            std::cout << duration << std::endl;
        }
    }
};

int Timer::cnt = 0;

