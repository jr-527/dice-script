'''Internal math functions'''
import numpy as np
import my_c_importer as my_c
import re

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
        if type(key) == slice:
            key = np.r_[key]
            return [self[i] for i in key]
        if self.start <= key and self.start + len(self.arr) > key:
            return self.arr[key-self.start]
        return 0.0

    def __repr__(self):
        return f'die({self.arr}, {self.start}, {self}, {self.basicName})'

    def __str__(self):
        if self.basicName:
            return self.name
        return f'({self.name})'

    def __truediv__(self, other):
        '''
        Gives the distribution of taking a sample from self and
        dividing by other, rounding towards 0.
        '''
        if isinstance(other, die):
            if other[0] >= 2**(-53):
                raise ZeroDivisionError("Cannot divide by a distribution that's sometimes 0")
            return divide_pmfs(self, other)
            # return NotImplemented
        if not is_number(other):
            raise TypeError('Can only divide by numbers')
        if other == 0:
            raise ZeroDivisionError('Cannot divide by zero')
        if other < 0:
            return (-self)/(-other)
        if other == 1:
            return self
        new_start = int(np.trunc(self.start/other))
        out = my_c.divide_pmf_by_int(self.arr, self.start, other)
        return die(out, new_start, f'{self}/{other}', True)

    def __floordiv__(self, other):
        '''
        Equivalent to __truediv__, so self/other.
        '''
        return self/other

    def _equals(self, other):
        '''
        Returns True if self and other have the same distribution, False otherwise.
        Note that this uses np.isclose to check equality.
        '''
        if self is other:
            return True
        if not isinstance(other, die):            
            if len(self.arr) == 1:
                return self.start == other
            raise ValueError(f'Invalid comparison of {self} and {other}')
        return (self.start == other.start and len(self.arr) == len(other.arr) and
            np.all(np.isclose(self.arr, other.arr)))

    def _comparison(self, relation, other):
        if not (isinstance(other, die) or is_number(other)):
            if any((isinstance(element, die) for element in other)):
                return NotImplemented
            other = set(other)
            return np.sum(((self==element)[1] for element in other))
        other_arr = None
        other_start = other
        if is_number(other):
            other_arr = np.array([1.0])
        elif isinstance(other, die):
            other_arr = other.arr
            other_start = other.start
        else:
            return NotImplemented
        relations = {'>':np.greater, '<':np.less, '>=':np.greater_equal,
                     '<=':np.less_equal, '==':np.equal, '!=':np.not_equal}
        prod = np.outer(self.arr, other_arr)
        indices = np.indices(prod.shape)
        bools = relations[relation](
            indices[0],
            indices[1]+(other_start-self.start)
        )
        out = np.sum(prod*bools)
        if PRINT_COMPARISONS[0]:
            print(f'P[{self} {relation} {other}] = {out}')
        return out

    def __mul__(self, other):
        '''
        Gives the distribution of the product of self, other.
        other: A number or die class object.
        Returns a new die class object.
        '''
        if isinstance(other, die):
            return multiply_pmfs(self, other)
        if not is_number(other):
            return NotImplemented
        if 0 < abs(other) and abs(other) < 1:
            return self / (1/other)
        other = round(other)
        if other == 0:
            return die([1.0], 0, '0', True)
        if other == 1:
            return self
        if other < 0:
            return -self*abs(other)
        arr = np.zeros(len(self.arr)*other)
        i = np.arange(len(self.arr))*other
        np.put(arr, i, self.arr)
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
            x = np.append(self.arr, [0]*(len(self.arr)-1)*(n-1))
            x = np.fft.irfft(np.fft.rfft(x)**n, len(x))
            start = n*self.start
            return die(x, n*self.start, f'{other}@{self}', False)
        if isinstance(other, die):
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
                # Could reuse forward ffts here but it shouldn't matter
                temp = other @ (i + ss)
                out_arr[temp.start-min_a:temp.start-min_a+len(temp.arr)] += p*temp.arr
            return die(out_arr, min_a, f'{self} @ {other}', False)
        return NotImplemented

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
        if not is_number(n):
            return NotImplemented
        if n != round(n) or n < 0:
            raise ValueError('Can only raise die to non-negative integer power')
        n = round(n)
        if n == 0:
            return die([1.0], 0, 1)
        if n == 1:
            return self
        ss = self.start
        se = self.start + len(self.arr)
        out = np.zeros(se**n-ss**n)
        i = (np.arange(len(self.arr))+ss)**n - ss**n
        np.put(out, i, self.arr)
        return die(out, min(ss**n, se**n), f'{self}^{n}', self.basicName)

    def __add__(self, other):
        '''
        Returns the distribution of the sum of self and other.
        other: A number or another die class object.
        Returns a new die class object.
        '''
        if isinstance(other, die):
            start = self.start + other.start
            arr = my_convolve(self.arr, other.arr)
            return die(arr, start, f'{self}+{other}', False)
        if not is_number(other):
            return NotImplemented
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
        p = self._comparison('==', other)
        if p is NotImplemented:
            return NotImplemented
        return die([1-p,p], 0, f'[{self} = {other}]', isProbability=True)

    def __ne__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self != other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        p = self._comparison('!=', other)
        if p is NotImplemented:
            return NotImplemented
        return die([1-p,p], 0, f'[{self} != {other}]', isProbability=True)

    def __lt__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self < other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        p = self._comparison('<', other)
        if p is NotImplemented:
            return NotImplemented
        return die([1-p,p], 0, f'[{self} < {other}]', isProbability=True)
    
    def __le__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self <= other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        p = self._comparison('<=', other)
        if p is NotImplemented:
            return NotImplemented
        return die([1-p,p], 0, f'[{self} <= {other}]', isProbability=True)

    def __gt__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self > other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        p = self._comparison('>', other)
        if p is NotImplemented:
            return NotImplemented
        return die([1-p,p], 0, f'[{self} > {other}]', isProbability=True)

    def __ge__(self, other):
        '''
        UNUSUAL BEHAVIOR - DOES NOT RETURN A BOOLEAN VALUE (more useful this way)

        Calculates the probability that self >= other, returns a Bernoulli distribution
        that's 1 with that probability, 0 otherwise.
        other: A die class object or a number.
        Returns a new die class object.
        '''
        p = self._comparison('>=', other)
        if p is NotImplemented:
            return NotImplemented
        return die([1-p,p], 0, f'[{self} >= {other}]', isProbability=True)

    def __bool__(self):
        if np.isclose(self[0], 1.0):
            return False
        if np.isclose(self[1], 1.0):
            return True
        if self.isProbability:
            raise ValueError('Cannot convert uncertain probability to boolean or evaluate expressions like 3 < 2d4 < 5.')
        return True

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

def divide_pmfs(x, y):
    '''
    Internal function, returns the distribution of the ratio of two RVs.
    x: A die class object, the numerator
    y: A die class object, the denominator
    Returns a new die class object.
    '''
    out, out_start = my_c.divide_pmfs(x.arr, y.arr, x.start, y.start)
    return die(out, out_start, f'{x}/{y}', True)

def multiply_pmfs(x, y):
    '''
    Internal function, returns the distribution of the product of two RVs.
    x: A die class object
    y: A die class object
    Returns a new die class object.
    '''
    # I can't find an efficient way to do this, so we'll do it the slow way, but in C.
    # It might be possible to do something involving characteristic functions,
    # but I'm not quite seeing it.
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
    