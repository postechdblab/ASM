#include <string>
#include <iostream>
#include <omp.h>
#include <regex>
#include <assert.h>

using namespace std;

// While using a single thread does not increase the inference time that much (about 2x increase),
// we used 32 threads to mimic the vectorized operation of numpy.
// Still, the following code takes the most of inference time.
// The parallel vector operation using openmp will be replaced by code that utilizes GPU.
// This will further enhance the inference time.

extern "C" {

    void string_contains(int n, char* c_strs[], char* c_substr, bool* output) {
        omp_set_num_threads(32);
        std::cout << "string_contains called! " << omp_get_num_threads() << " " << omp_get_max_threads() << std::endl;
        const std::string substr(c_substr);
        std::cout << "substr " << substr << std::endl;
#pragma omp parallel 
        {
            int nt = omp_get_num_threads();
            int t  = omp_get_thread_num();
            int s  = (t+0) * n / nt;
            int f  = (t+1) * n / nt;

            for (int i = s; i < f; i++) {
                if (c_strs[i]) {
                    const std::string str(c_strs[i]);
                    output[i] = str.find(substr) != std::string::npos;
                }
                else {
                }
            }
        }
    }

    void pattern_match(int n, char* c_strs[], char* c_pattern, bool* output) {
        omp_set_num_threads(32);
        const std::string pattern(c_pattern);
        std::regex reg(pattern);
#pragma omp parallel 
        {
            int nt = omp_get_num_threads();
            int t  = omp_get_thread_num();
            int s  = (t+0) * n / nt;
            int f  = (t+1) * n / nt;

            if (t == 0)
                assert(s == 0);
            else if (t == omp_get_num_threads() - 1)
                assert(f == n);

            for (int i = s; i < f; i++) {
                if (c_strs[i]) {
                    const std::string str(c_strs[i]);
                    output[i] = std::regex_match(str, reg); 
                }
                else {
                    output[i] = false;
                }
            }
        }
    }
}
