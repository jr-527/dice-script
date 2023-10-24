__declspec(dllexport)
void multiply_pmfs(double* out, double* x, int len_x, double* y, int len_y,
                   int x_min, int y_min, int lower_bound) {
    int j = 0;
    int index = 0;
    for (int i = 0; i < len_x; i++) {
        for (j = 0; j < len_y; j++) {
            index = (i+x_min)*(j+y_min)-lower_bound;
            out[index] += x[i]*y[j];
        }
    }
}

__declspec(dllexport)
void pmf_times_int(double* out, double* in, int len_in, int factor) {
    for (int i = 0; i < len_in; i++) {
        out[i*factor] = in[i];
    }
}