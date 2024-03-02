import ctypes as ct
import numpy as np
import sys, os

# python -> c:
# ct.c_int(x)
# ct.c_size_t(x)
# ct.c_double(x)
# ct.POINTER(ct.c_double)
# c -> python:
# int(x)

d_arr = ct.POINTER(ct.c_double)

def arr_to_c(arr):
    return np.ctypeslib.as_ctypes(np.array(arr))

try:
    lib_path = ''
    if os.name == 'nt':
        lib_path = 'helpers.dll'
    elif os.name == 'posix':
        lib_path = 'helpers.so'
    dll = ct.CDLL(os.path.join(sys.path[0], lib_path))
    dll.multiply_pmfs.argtypes = (
        d_arr,
        d_arr, ct.c_size_t,
        d_arr, ct.c_size_t, 
        ct.c_size_t, ct.c_size_t, ct.c_size_t
    )
    dll.divide_indices.argtypes = (
        d_arr,
        d_arr, ct.c_int64,
        ct.c_double, ct.c_int64
    )

    def divide_pmf_by_int(arr, start, n):
        global dll, arr_to_c
        end = start + len(arr)
        out = np.zeros(int(np.trunc(end/n))-int(np.trunc(start/n))+1, np.dtype('d'))
        out_c = np.ctypeslib.as_ctypes(out)
        arr_c = arr_to_c(arr)
        dll.divide_indices(
            out_c,
            arr_c, ct.c_int64(len(arr)),
            ct.c_double(n), ct.c_int64(start)
        )
        return out

    def multiply_pmfs(arr, x, y, x_min, y_min, lower_bound):
        global dll, arr_to_c
        arr_c = arr_to_c(arr)
        x_c = arr_to_c(x)
        y_c = arr_to_c(y)
        dll.multiply_pmfs(
            arr_c,
            x_c, ct.c_size_t(len(x)),
            y_c, ct.c_size_t(len(y)),
            ct.c_size_t(x_min), ct.c_size_t(y_min), ct.c_size_t(lower_bound)
        )
        return np.ctypeslib.as_array(arr_c, len(arr))
except:
    print('Error importing C functions, using Python fall-back')
    def multiply_pmfs(arr, x, y, x_min, y_min, lower_bound):
        for i in range(len(x)):
            for j in range(len(y)):
                index = (i+x_min)*(j+y_min)-lower_bound
                arr[index] += x[i]*y[j]
        return arr

    def divide_pmf_by_int(arr, start, n):
        end = start + len(arr)
        new_start = int(np.trunc(start/n))
        new_end = int(np.trunc(end/n))
        out = np.zeros(new_end-new_start+1, np.dtype('d'))
        for i, val in enumerate(arr):
            out[int(np.trunc((start+i)/n))-new_start] += val
        return out
