import ctypes as ct
import numpy as np
import sys, os

# python -> c:
# ct.c_int(x)
# ct.c_double(x)
# ct.POINTER(ct.c_double)
# c -> python:
# int(x)

d_arr = ct.POINTER(ct.c_double)

def x_to_c(x):
    if isinstance(x, (int, np.integer)):
        return ct.c_int(x)
    if isinstance(x, (float, np.double)):
        return ct.c_double(x)
    print(x, type(x))
    raise TypeError("Must be int or floating point value")

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
        d_arr, ct.c_int,
        d_arr, ct.c_int, 
        ct.c_int, ct.c_int, ct.c_int
    )
    dll.pmf_times_int.argtypes = (d_arr, d_arr, ct.c_int, ct.c_int)

    def multiply_pmfs(arr, x, y, x_min, y_min, lower_bound):
        global dll, arr_to_c, x_to_c
        arr_c = arr_to_c(arr)
        x_c = arr_to_c(x)
        y_c = arr_to_c(y)
        x_min = x_to_c(x_min)
        y_min = x_to_c(y_min)
        lower_bound = x_to_c(lower_bound)
        dll.multiply_pmfs(
            arr_c,
            x_c, ct.c_int(len(x)),
            y_c, ct.c_int(len(y)),
            x_min, y_min, lower_bound
        )
        return np.ctypeslib.as_array(arr_c, len(arr))

    def pmf_times_int(pmf, n):
        out = np.ctypeslib.as_ctypes(np.zeros(len(pmf) * n))
        dll.pmf_times_int(out, np.ctypeslib.as_ctypes(pmf), x_to_c(len(pmf)), x_to_c(n))
        return np.ctypeslib.as_array(out, len(pmf)*n)
except:
    print('Error importing C functions, using Python fall-back')
    def multiply_pmfs(arr, x, y, x_min, y_min, lower_bound):
        for i in range(len(x)):
            for j in range(len(y)):
                index = (i+x_min)*(j+y_min)-lower_bound
                arr[index] += x[i]*y[j]
        return arr

    def pmf_times_int(pmf, n):
        out = [0.0] * len(pmf)*n
        for i, val in enumerate(pmf):
            out[i*n] = pmf[i]
        return out
