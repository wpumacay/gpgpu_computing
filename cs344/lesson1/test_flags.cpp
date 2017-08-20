

#include <iostream>

// #define USE_CUDA 1
// #define USE_OPENCL 1


using namespace std;

int main()
{
    #ifdef USE_CUDA

    cout << "cuda version" << endl;

    #elif USE_OPENCL
    
    cout << "opencl version" << endl;

    #else

    cout << "no version selected" << endl;

    #endif


    return 0;
}
