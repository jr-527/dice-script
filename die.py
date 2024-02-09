'''Internal math functions'''
import numpy as np
import my_c_importer as my_c
import re
# warnings.filterwarnings('ignore', 'elementwise comparison failed')

PRINT_COMPARISONS = [False]

class die:
    '''
    A class for managing PMFs which take on integer values. A number of built-in functions like
    __mul__ are defined, so if x, y, z are instances of this class, you can do things like (x+y)*z.
    Instances of this class are immutable, so all methods return an object and none are done in-place.
    
    Initialization parameters:
    arr: A list or numpy array that's a valid PMF (non-negative numbers which sum to 1)
    start: An integer, the offset for the first non-zero value in arr,
           so die([.4,.6], 5) is a 40% chance of 5, 60% chance of 6.
    name (optional): A string, a name for this distribution
    basicName (optional): Boolean, should the name be parenthesized when combining with other names.
    '''
    # Immutable. All methods should return an object.
    # Invariant: self.arr is a valid PMF
    def __init__(self, arr, start, name=None, basicName=False, isProbability=False):
        '''
        arr: A list or numpy array that's a valid PMF (non-negative numbers which sum to 1)
        start: An integer, the offset for the first non-zero value in arr,
               so die([.4,.6], 5) is a 40% chance of 5, 60% chance of 6.
        name (optional): A string, a name for this distribution
        basicName (optional): Boolean, should the name be parenthesized when combining with other names.
        '''
        self.start, self.arr = trim(arr)
        self.start += start
        self.name = re.sub('\+\-', '-', str(name))
        self.name = re.sub('\-\-', '+', self.name)
        self.isProbability = isProbability
        if len(self.name) > 0 and self.name[0] == '+':
            self.name = self.name[1:]
        self.basicName = basicName

    def __getitem__(self, key):
        if self.start <= key and self.start + len(self.arr) > key:
            return self.arr[key-self.start]
        return 0.0

    def __repr__(self):
        return f'die({self.arr}, {self.start}, {self}, {self.basicName})'

    def __str__(self):
        if self.basicName:
            return self.name
        return f'({self.name})'

    def __truediv__(self, n):
        '''
        Gives the distribution of taking a sample from self and dividing by n, rounding
        towards 0.
        '''
        if not is_number(n):
            raise TypeError('Can only divide by numbers')
        if n == 0:
            return 1/0 # gives the right error
        if n < 0:
            return (-self)/(-n)
        if n == 1:
            return self
        ss = self.start
        new_start = int_div_to_0(ss, n)
        se = self.start + len(self.arr)
        new_end = int_div_to_0(se, n)
        out = [0.0]*(new_end-new_start+1)
        j = 0
        for i, val in enumerate(self.arr):
            j = int_div_to_0(ss+i,n) - new_start
            out[j] += val
        return die(out, new_start, f'{self}/{n}', True)

    def __floordiv__(self, n):
        '''
        Equivalent to __truediv__, so self/n.
        '''
        return self/n

    def equals(self, other):
        '''
        Returns True if self and other have the same distribution, False otherwise.
        Note that this uses np.isclose to check equality.
        '''
        if self is other:
            return True
        if type(self) != type(other):
            if len(self.arr) == 1:
                return self.start == other
            raise ValueError(f'Invalid comparison of {self} and {other}')
        return (self.start == other.start and len(self.arr) == len(other.arr) and
            np.all(np.isclose(self.arr, other.arr)))

    def __mul__(self, other):
        '''
        Gives the distribution of the product of self, other.
        other: A number or die class object.
        Returns a new die class object.
        '''
        if type(other) == type(self):
            return multiply_pmfs(self, other)
        if 0 < abs(other) and abs(other) < 1:
            return self / (1/other)
        other = round(other)
        if other == 0:
            return die([1.0], 0, '0', True)
        if other == 1:
            return self
        if other < 0:
            return -self*abs(other)
        arr = my_c.pmf_times_int(self.arr, other)
        return die(arr, self.start*other, f'{self}*{other}', True)

    def __matmul__(self, other):
        '''
        Gives the distribution of sampling from self, then summing up that many samples from other.
        Uses FFT wherever possible.
        Note that 2 @ 1d4 == 2d4 != 2*1d4.
        Returns a new die class object.
        '''
        if is_number(other):
            if other < 0:
                return -(self @ -other)
            if other == 0:
                return self*0
            n = round(other)
            # out = 0
            # latest = self
            # mask = 1
            # while mask <= n:
            #     if n & mask:
            #         out += latest
            #     latest += latest
            #     mask *= 2
            x = np.append(self.arr, [0]*(len(self.arr)-1)*(n-1))
            x = np.fft.irfft(np.fft.rfft(x)**n, len(x))
            start = n*self.start
            return die(x, n*self.start, f'{other}@{self}', False)
        if type(other) == type(self):
            ss = self.start
            se = self.start + len(self.arr)
            os = other.start
            oe = other.start + len(other.arr)
            min_a = min(ss*os, ss*oe, se*os, se*oe)
            max_a = max(ss*os, ss*oe, se*os, se*oe)
            out_arr = np.array([0.0] * (max_a - min_a + 1))
            out = die([1.0], 0, 1)
            for i, p in enumerate(self.arr):
                # This is the equivalent of doing
                # out = [0, ..., 0]
                # for offset in offsets:
                #     out += original * offset
                # return out
                temp = other @ (i + ss)
                # I could optimize the previous line, because it's needlessly re-calculating
                # the forward FFT of other with each iteration of this loop, but it would
                # be a ton of work and wouldn't quite halve the run time of this function,
                # so I won't bother.
                # x = np.concatenate( # np.concatenate is unwieldy compared to R's c()
                #     (
                #         [0.0] * (temp.start-min_a),
                #         temp.arr*p,
                #         [0.0] * (1+max_a-(temp.start+len(temp.arr)))
                #     ),
                #     axis = None
                # )
                # out_arr = out_arr + x
                out_arr[temp.start-min_a:temp.start-min_a+len(temp.arr)] += p*temp.arr
            return die(out_arr, min_a, f'{self} @ {other}', False)

    def __rmatmul__(self, other):
        '''Variant of __matmul__, self-explanatory.'''
        return self @ other

    def __rmul__(self, other):
        '''Variant of __mul__, self-explanatory.'''
        return self*other

    def __neg__(self):
        '''
        Returns the distribution of the negative of self.
        '''
        out_basic_name = not self.basicName
        if re.fullmatch('(\-)?[1-9][0-9]*d[1-9][0-9]*', self.name):
            out_basic_name = True
        return die(np.flip(self.arr), -self.start-len(self.arr)+1, f'-{self}', out_basic_name)

    def __pow__(self, n):
        '''
        Returns the distribution of taking a sample from self and raising the result to
        the nth power.
        n: A non-negative integer
        Returns a new die class object.
        '''
        if not is_number(n) or n != round(n) or n < 0:
            raise TypeError('Can only raise a die to a non-negative integer power')
        n = round(n)
        if n == 0:
            return die([1.0], 0, 1)
        if n == 1:
            return self
        ss = self.start
        se = self.start + len(self.arr)
        out = [0.0] * (se**n - ss**n)
        for i, val in enumerate(self.arr):
            out[(i+ss)**n - ss**n] = val
        return die(out, min(ss**n, se**n), f'{self}^{n}', self.basicName)

    def __add__(self, other):
        '''
        Returns the distribution of the sum of self and other.
        other: A number or another die class object.
        Returns a new die class object.
        '''
        if type(other) == type(self):
            start = self.start + other.start
            arr = my_convolve(self.arr, other.arr)
            return die(arr, start, f'{self}+{other}', False)
        other = round(other)
        return die(self.arr, self.start+other, f'{self}+{other}', False)

    def __radd__(self, other):
        '''Variant of __add__, self-explanatory'''
        return self+other

    def __sub__(self, other):
        '''Variant of __add__, self-explanatory'''
        return self + (-other)

    def __rsub__(self, other):
        '''Variant of __add__, self-explanatory'''
        return -self + other

    def __eq__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self == other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        if is_number(other):
            a = self.arr
            a = a[np.indices(a.shape)[0] + self.start == other]
            s = np.sum(a)
            if PRINT_COMPARISONS[0]:
                print(f'P[{self} = {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s], 0, f'[{self} = {other}]', isProbability=True)
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start == 0]
        s = np.sum(a)
        if PRINT_COMPARISONS[0]:
            print(f'P[{self} = {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s], 0, f'[{self} = {other}]', isProbability=True)
        # return self

    def __lt__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self < other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        if is_number(other):
            a = self.arr
            a = a[np.indices(a.shape)[0]+self.start < other]
            s = np.sum(a)
            if PRINT_COMPARISONS[0]:
                print(f'P[{self} < {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s], 0, f'[{self} < {other}]', True, isProbability=True)
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start < 0]
        s = np.sum(a)
        if PRINT_COMPARISONS[0]:
            print(f'P[{self} < {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s], 0, f'[{self} < {other}]', True, isProbability=True)
        # return self

    
    def __le__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self <= other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        if is_number(other):
            a = self.arr
            a = a[np.indices(a.shape)[0]+self.start <= other]
            s = np.sum(a)
            if PRINT_COMPARISONS[0]:
                print(f'P[{self} <= {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s], 0, f'[{self} <= {other}]', True, isProbability=True)
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start <= 0]
        s = np.sum(a)
        if PRINT_COMPARISONS[0]:
            print(f'P[{self} <= {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s], 0, f'[{self} <= {other}]', True, isProbability=True)
        # return self

    def __gt__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self > other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        if is_number(other):
            a = self.arr
            a = a[np.indices(a.shape)[0]+self.start > other]
            s = np.sum(a)
            if PRINT_COMPARISONS[0]:
                print(f'P[{self} > {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s], 0, f'[{self} > {other}]', True, isProbability=True)
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start > 0]
        s = np.sum(a)
        if PRINT_COMPARISONS[0]:
            print(f'P[{self} > {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s], 0, f'[{self} > {other}]', True, isProbability=True)
        # return self

    def __ge__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self >= other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        if is_number(other):
            a = self.arr
            a = a[np.indices(a.shape)[0]+self.start >= other]
            s = np.sum(a)
            if PRINT_COMPARISONS[0]:
                print(f'P[{self} >= {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s], 0, f'[{self} >= {other}]', True, isProbability=True)
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start >= 0]
        s = np.sum(a)
        if PRINT_COMPARISONS[0]:
            print(f'P[{self} >= {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s], 0, f'[{self} >= {other}]', True, isProbability=True)

def ndm(n, m):
    '''
    Internal function, returns the distribution of ndm. Uses an FFT-based algorithm for the calculations
    Ex: ndm(3, 6) returns the distribution of 3d6.
    n: A positive integer
    m: A positive integer
    Returns a die class object.
    '''
    x = [1.0]*m + [0.0]*m*(n-1)
    if n == 1:
        return np.array(x)/m
    out = None
    try:
        f = float(m**n)
        out = np.rint(np.fft.irfft(np.fft.rfft(x)**n, len(x))) / f
    except OverflowError:
        out = np.fft.irfft(np.fft.rfft(x/np.sum(x))**n, len(x))
    return out[:-(n-1)]

def is_number(x):
    '''Internal function. Returns True if x is a numeric type, False otherwise.'''
    return isinstance(x, (int, float, np.number))

def int_div_to_0(x, n):
    '''
    Internal function. Equivalent to x//n except negative values round towards 0.
    '''
    if x < 0:
        return int(-(-x//n))
    return int(x//n)

def trim(arr):
    '''
    Internal function, trims trailing and leading zeros from arr.
    Ex: trim([0.0, 0.0, 1.0, 0.0, 1.1, 0.0]) returns [1.0, 1.1]
    arr: list or numpy array
    Returns a numpy array.
    '''
    first, last = 0, len(arr)
    for i in arr:
        if i != 0.:
            break
        else:
            first += 1
    for i in arr[::-1]:
        if i != 0.:
            break
        else:
            last -= 1
    return first, np.array(arr[first:last])

def pad(arr, start, length):
    '''
    Internal function, pads an array with zeros so that the old start is at index start,
    and so that the total length is length. length must be long enough.
    arr: A list or numpy array
    start: an integer
    length: an integer
    Returns a numpy array.
    '''
    start = round(start)
    x = arr
    if start < 0:
        raise ValueError('pad(): start must be >= 0')
    if start > 0:
        x = np.append([0.0]*start, x)
    return np.append(x, [0.0]*(length-len(x)))

def multiply_pmfs(x, y):
    '''
    Internal function, returns the distribution of the product of two RVs.
    x: A die class object
    y: A die class object
    Returns a new die class object.
    '''
    # I can't find an efficient way to do this, so we'll do it the slow way, but in C.
    # https://en.wikipedia.org/wiki/Distribution_of_the_product_of_two_random_variables
    if len(x.arr) > len(y.arr):
        x, y = y, x
    x_min, y_min = x.start, y.start
    x_max = x_min + len(x.arr)
    y_max = y_min + len(y.arr)
    temp = np.outer((x_min, x_max), (y_min, y_max))
    lower_bound = np.min(temp)
    upper_bound = np.max(temp)
    arr = [0.0]*(upper_bound-lower_bound+1)
    arr = my_c.multiply_pmfs(arr, x.arr, y.arr, x_min, y_min, lower_bound)
    return die(arr, lower_bound, f'{x}*{y}', True)

def bin_coeff(n, k):
    '''Internal function, returns the binomial coefficient.'''
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n-k))

def my_convolve(x, y):
    '''
    Internal function for FFT convolutions.
    x, y: Input lists/arrays
    Returns the convolution of x, y
    '''
    n = len(x)
    m = len(y)
    x = np.append(x, [0]*m)
    y = np.append(y, [0]*n)
    convolve = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(y), n+m)
    return convolve[:-1]