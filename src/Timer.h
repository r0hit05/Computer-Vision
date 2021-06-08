#pragma once
#include <chrono>
#include <iostream>

using namespace std;

struct Timer
{
	std::chrono::steady_clock::time_point start; 
	float mtplr;
	static int cnt;

	Timer(float f = 1)
		: mtplr(f)
	{
		cnt++;
		start = std::chrono::high_resolution_clock::now();
	}

	~Timer()
	{
		std::chrono::duration<float> duration = std::chrono::high_resolution_clock::now() - start;
		cnt--;
		float time = duration.count() * mtplr;
		if (!cnt)
		{
			std::cout << time;

			if (mtplr == 1000.0)
				std::cout << "m";
			else if (mtplr == 1000000.0)
				std::cout << "mu";

			std::cout << "s" << std::endl;
		}
			
	}

};

int Timer::cnt = 0;

