// MSC compilation: cl /O2 /D_USRDLL /D_WINDLL helpers.c /link /DLL /OUT:helpers.dll
// GCC compilation: gcc -O3 -shared -o helpers.so -fPIC helpers.c
#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
__declspec(dllexport)
#endif
void multiply_pmfs(double* out, double* x, size_t len_x, double* y, size_t len_y,
                   size_t x_min, size_t y_min, size_t lower_bound) {
    size_t j = 0;
    size_t index = 0;
    for (size_t i = 0; i < len_x; i++) {
        for (j = 0; j < len_y; j++) {
            index = (i+x_min)*(j+y_min)-lower_bound;
            out[index] += x[i]*y[j];
        }
    }
}

#ifdef _MSC_VER
__declspec(dllexport)
#endif
void divide_indices(double* out, double* x, int64_t len_x, double n, int64_t start) {
    int64_t new_start = (int64_t)(start/n);
    for (int64_t i = 0; i < len_x; i++) {
        out[(int64_t)((start+i)/n) - new_start] += x[i];
    }
}