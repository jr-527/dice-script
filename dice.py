import os
import sys
import numpy as np
plt = None # For asynchronous importing, to start up more quickly
import threading
import re
import warnings
import my_c_importer as my_c
import ctypes as ct
import traceback
warnings.filterwarnings('ignore', 'elementwise comparison failed')

plt_initialized = False
def import_plt():
    global plt
    global plt_initialized
    import matplotlib.pyplot as plt
    plt_initialized = True
import_thread = threading.Thread(target=import_plt, name='import matplotlib')
import_thread.start()
ct.windll.kernel32.SetConsoleTitleW('Dice Script')

PRINT_COMPARISONS = True

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
    def __init__(self, arr, start, name=None, basicName=False):
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
        if len(self.name) > 0 and self.name[0] == '+':
            self.name = self.name[1:]
        self.basicName = basicName

    def __getitem__(self, value):
        if self.start <= value and self.start + len(self.arr) > value:
            return self.arr[value-self.start]
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
            x = pad(self.arr, self.start, n*(self.start + len(self.arr)))
            out = np.fft.irfft(np.fft.rfft(x)**n, len(x))
            # x = pad(self.arr, self.start, n*(self.start + len(self.arr))) * self.denominator
            # out = np.rint(np.fft.irfft(np.fft.rfft(x)**n, len(x))) / (self.denominator**n)
            start, arr = trim(out)
            # we correct for arr still sometimes being too long
            max_a = n*(self.start+len(self.arr)-1)
            actual_a = start + len(arr) - 1
            if max_a < actual_a:
                arr = arr[:max_a - actual_a]
            return die(arr, start, f'{other} @ {self}', False)
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
                if temp.start != other.start*(i+ss):
                    # This handles weird floating point rounding issues, I think.
                    diff = other.start*(i+ss) - temp.start
                    temp.arr = temp.arr[diff:]
                    temp.start += diff
                x = np.concatenate( # np.concatenate is unwieldy compared to R's c()
                    (
                        [0.0] * (temp.start-min_a),
                        temp.arr*p,
                        [0.0] * (1+max_a-(temp.start+len(temp.arr)))
                    ),
                    axis = None
                )
                out_arr = out_arr + x
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
            if PRINT_COMPARISONS:
                print(f'P[{self} = {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s],0,f'[{self} = {other}]')
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start == 0]
        s = np.sum(a)
        if PRINT_COMPARISONS:
            print(f'P[{self} = {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s],0,f'[{self} = {other}]')
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
            if PRINT_COMPARISONS:
                print(f'P[{self} < {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s],0,f'[{self} < {other}]', True)
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start < 0]
        s = np.sum(a)
        if PRINT_COMPARISONS:
            print(f'P[{self} < {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s],0,f'[{self} < {other}]', True)
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
            if PRINT_COMPARISONS:
                print(f'P[{self} <= {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s],0,f'[{self} <= {other}]', True)
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start <= 0]
        s = np.sum(a)
        if PRINT_COMPARISONS:
            print(f'P[{self} <= {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s],0,f'[{self} <= {other}]', True)
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
            if PRINT_COMPARISONS:
                print(f'P[{self} > {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s],0,f'[{self} > {other}]', True)
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start > 0]
        s = np.sum(a)
        if PRINT_COMPARISONS:
            print(f'P[{self} > {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s],0,f'[{self} > {other}]', True)
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
            if PRINT_COMPARISONS:
                print(f'P[{self} >= {other}] =', np.format_float_positional(s,14,trim='-'))
            return die([1-s,s],0,f'[{self} >= {other}]', True)
            # return self
        t = self-other
        a = t.arr
        a = a[np.indices(a.shape)[0]+t.start >= 0]
        s = np.sum(a)
        if PRINT_COMPARISONS:
            print(f'P[{self} >= {other}] =', np.format_float_positional(s,14,trim='-'))
        return die([1-s,s],0,f'[{self} >= {other}]', True)

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
    out = np.rint(np.fft.irfft(np.fft.rfft(x)**n, len(x))) / float(m**n)
    return out[:-(n-1)]

def process_input(text):
    '''
    Internal function. Processes then calls eval on text, plotting if possible.
    '''
    new_text = re.sub('\s', ' ', text)
    new_text = re.sub(r'[)]d', r')@1d', new_text)
    new_text = re.sub('(4d6dl)|(4d6 drop lowest)', 'stat_roll()', new_text)
    new_text = re.sub(r'\^', '**', new_text)
    new_text = re.sub(r'([1-9][0-9]*)d([1-9][0-9]*)',
        r'die(ndm(int(\1),int(\2)),int(\1),"\1d\2", True)', new_text)
    x = eval(new_text)
    if x is None:
        print('Nothing to plot.')
    else:
        plot(x, text)

def min0(d):
    '''
    Transforms d so that values less than 0 are increased to 0.
    d: A die class object
    Returns a new die class object.
    '''
    return min_val(d, 0)

def min1(d):
    '''
    Transforms d so that values less than 1 are increased to 1.
    d: A die class object
    Returns a new die class object.
    '''
    return min_val(d, 1)

def min_val(d, m=1):
    '''
    Transforms a distribution so that values less than m are increased to m.
    d: A die class object.
    m: an integer, the new minimum
    Returns a new die class object.
    '''
    if d.start >= m:
        return d
    if d.start + len(d.arr) < m:
        return die([1.0], m, 1)
    s = np.sum(d.arr[:-d.start+m])
    out = np.copy(d.arr[-d.start+m:])
    out[0] += s
    out_str = ''
    if m == 0:
        if d.basicName:
            out_str = f'min0({d})'
        else:
            out_str = f'min0{d}'
    if m == 1:
        if d.basicName:
            out_str = f'min1({d})'
        else:
            out_str = f'min1{d}'
    else:
        if d.basicName:
            out_str = f'min_val({d}, {m})'
        else:
            out_str = f'min_val{d}'
            out_str = out_str[:-1] + f', {m})'
    return die(out, m, out_str)

def mean(d):
    '''Internal function, returns the mean of a die class object.'''
    x = np.arange(d.start, d.start+len(d.arr))
    return np.sum(x * d.arr)

def var(d):
    '''Internal function, returns the variance of a die class object.'''
    x = np.arange(d.start, d.start+len(d.arr))
    mu = mean(d)
    return max(np.sum(d.arr * (x-mu)**2), 0.0)

def sd(d):
    '''Internal function, returns the standard deviation of a die class object.'''
    return np.sqrt(var(d))

def plot(d, name=None):
    '''
    Internal function. Plots a distribution
    d: A number or die class object
    name (optional): A custom name for the plot.
    '''
    global plt_initialized
    # If matplotlib isn't imported yet, we wait
    if not plt_initialized:
        import_thread.join()
        plt_initialized = True
    if is_number(d):
        d = round(d)
        d = die([1], d, 1)
    print(f'Mean: {round(mean(d),4)}, standard deviation: {round(sd(d),4)}')
    print(f'Random sample from distribution: {sample(d)}')
    fig, ax = plt.subplots()
    if name:
        if '=' in name or '<' in name or '>' in name:
            name = d.name
        fig.canvas.manager.set_window_title(name)
        plt.title(name)
    y = d.arr
    cumulative = np.cumsum(y)
    x = range(d.start, d.start+len(y))
    ax.stem(x, y, label='Probability', basefmt='')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax.twinx()
    ax2.set_ylim(-.05, 1.05)
    ax2.plot(x, cumulative, 'tab:red', label='Cumulative')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.legend()
    plt.show()

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

def order_stat(x, num=1, pos=1):
    '''
    Returns the distribution of the pos order statistic of num iid samples of x.
    pos=1 corresponds to the lowest sample, pos=num to the highest sample.
    x: A die class object
    num: An integer
    pos: An integer
    Returns a new die class object.
    '''
    if pos < 0:
        pos = num + pos + 1
    if num == 1 and pos == 1:
        return x
    elif pos == 1:
        return lowest(x, num)
    elif pos == num:
        return highest(x, num)
    x_cdf = np.cumsum(x.arr)
    x_pmf = x.arr
    n = num
    k = pos
    pmf = x_pmf*0
    for j in range(n-k+1):
        temp = (1-x_cdf)**j * x_cdf**(n-j) - (1-x_cdf+x_pmf)**j * (x_cdf - x_pmf)**(n-j)
        temp *= bin_coeff(n,j)
        pmf += temp
    return die(pmf, x.start, f'order({x}, {num}, {pos})', True)

order = order_stat

def highest(x, n=2):
    '''
    Returns the distribution of the greatest of n iid samples from x.
    x: A die class object.
    n: An integer
    Returns a new die class object.
    '''
    if n == 1:
        return x
    cdf = np.cumsum(x.arr)
    pmf = np.ediff1d(cdf**n)
    pmf = np.append(cdf[0]**2, pmf)
    out_str = ''
    if n == 2:
        if x.basicName:
            out_str = f'adv({x})'
        else:
            out_str = f'adv{x}'
    else:
        if x.basicName:
            out_str = f'highest({x}, {n})'
        else:
            out_str = f'highest({x})'
            out_str = out_str[-1] + f', {n})'
    return die(pmf, x.start, out_str, True)

def adv(x):
    '''
    Returns the distribution of sampling twice from x and keeping the greater sample.
    x: A die class object
    Returns a new die class object.
    '''
    return highest(x, 2)
advantage = adv

def lowest(x, n=2):
    '''
    Returns the distribution of the lowest of n iid samples from x.
    x: A die class object.
    n: An integer
    Returns a new die class object.
    '''
    if n == 1:
        return x
    cdf = np.cumsum(x.arr)
    pmf = np.ediff1d(1-(1-cdf)**n)
    pmf = np.append(1-(1-cdf[0])**n, pmf)
    out_str = ''
    if n == 2:
        if x.basicName:
            out_str = f'dis({x})'
        else:
            out_str = f'dis{x}'
    else:
        if x.basicName:
            out_str = f'lowest({x}, {n})'
        else:
            out_str = f'lowest({x})'
            out_str = out_str[-1] + f', {n})'
    return die(pmf, x.start, out_str, True)

def disadv(x):
    '''
    Returns the distribution of sampling twice from x and keeping the lower sample.
    x: An object of class die.
    '''
    return lowest(x, 2)
dis = disadv
disadvantage = disadv

def choice(condition, *args):
    '''
    Returns the distribution of choosing an argument based on a sample from condition.
    condition: A probability distribution. Can be a list of non-negative numbers that sums to 1 or a die class.
               Note that "boolean" die expressions, eg 1d20 > 3, are Bernoulli RVs, so distributions with length 2.
    *args: Distributions to simulate samples from based on the result of condition. Number of distributions must
           be equal to the length of condition. Can be integers or die class objects.
    Ex:
    choice([0.4, 0.4, 0.2], 1d4, 1d6, 2d4)
    Returns the distribution that's a 40% chance of 1d4, 40% of 1d6, and 20% chance of 2d4

    choice(adv(1d20+4) >= 15, 8d6/2, 8d6)
    Returns the distribution that's 8d6/2 if adv(1d20+4) < 15, 8d6 otherwise

    choice(1d4, 1, 3, 3, 7)
    Returns the distribution that's 1 with probability 25%, 3 with probability 50%, 7 with probability 25%.
    '''
    n = len(args)
    probs = []
    if not isinstance(condition, die):
        if is_number(condition):
            if 0 <= condition and condition <= 1:
                probs = [condition, 1-condition]
            else:
                raise ValueError('Numeric condition in choice() must be a probability')
        elif hasattr(condition, '__len__'):
            condition = np.array(condition)
            if condition.shape != (n,):
                raise ValueError('List-like condition in choice() has incorrect shape')
            if np.isclose(sum(condition),1) and np.all(condition >= 0):
                probs = condition
            else:
                raise ValueError('List-like condition in choice() must be a PMF')
        else:
            raise TypeError('Invalid arguments to choice()')
    else:
        probs = condition.arr
        if len(probs) != n:
            raise TypeError('Invalid number of arguments for choice()')
    lb = np.inf
    rb = -np.inf
    for thing in args:
        if is_number(thing):
            lb = min(lb, round(thing))
            rb = max(rb, round(thing))
        elif isinstance(thing, die):
            lb = min(lb, thing.start)
            rb = max(rb, thing.start+len(thing.arr))
        else:
            raise TypeError('Invalid argument type for choice()')
    if lb > rb:
        raise ValueError('Error in choice()')
    out = np.zeros(rb-lb+1)
    for thing, p in zip(args, probs):
        if is_number(thing):
            out[round(thing)-lb] += p
        else:
            padded = np.hstack([
                [0.0]*(thing.start-lb),
                thing.arr,
                [0.0]*(rb-(thing.start+len(thing.arr)-1))
            ])
            out += padded * p
    if not np.isclose(sum(out),1):
        raise ValueError('Error in choice(): result is not a PMF')
    out_string = f'choice({condition}'
    for arg in args:
        out_string += f', {arg}'
    out_string += ')'
    return die(out, lb, out_string, True)

def attack(bonus, ac, damage, damage_bonus=0, *, extra_dice=0, crit_range=20, adv=0, no_crit_damage=False):
    '''
    Returns the damage distribution of making an attack using DnD 5e rules.
    bonus: The attack's to-hit bonus. Can be a number or die object.
    ac: The target's armor class. Can be a number or die object.
    damage: The attack's damage dice, so the part that's doubled on crits.
    damage_bonus: The attack's damage bonus, so the part that isn't doubled on
                  crits.
    extra_dice: Additional damage to add on crits.
    crit_range: The attack crits if the roll (not including bonuses) is at least
                crit_range.
    adv: 0 for a normal attack, True or 1 for advantage, -1 for disadvantage.
         2 or -2 for double adv/disadv, respectively, 3 or -3 for triple, etc.
    no_crit_damage: True changes crits to not deal extra damage, so they're just
                    guaranteed hits.
    '''
    global PRINT_COMPARISONS
    PRINT_COMPARISONS = False
    d20 = die(ndm(1,20), 1, '1d20', True)
    pos_val = 1 if (adv <= 0) else abs(adv)+1
    d20_roll = order_stat(d20, abs(adv)+1, pos_val)
    attack_roll = d20_roll + bonus
    attack_roll.basicName = True
    p_crit = (d20_roll >= crit_range)[1]
    # We want P(miss because natural 1)
    # =P(attack roll >= ac|nat 1)*P(nat 1)
    blocked_hit = (bonus+1) >= ac
    if not is_number(blocked_hit):
        ac.basicName = True
        blocked_hit = blocked_hit[1]
    p_relevant_nat1 = blocked_hit * (d20_roll == 1)[1]
    # the previous line is either a number or a die object
    # if it's a number, we're done. If it's a die object,
    # we need the following line to convert it into a number
    if not is_number(bonus):
        p_relevant_nat1 = p_relevant_nat1[1]
    p_regular_hit = (attack_roll >= ac)[1] - p_crit - p_relevant_nat1
    # In some cases a regular hit is impossible, ie AC 100
    p_regular_hit = max(0.0, p_regular_hit)
    p_miss = 1.0-p_regular_hit-p_crit
    regular_dmg = damage + damage_bonus
    regular_dmg.basicName = True
    crit_dmg = None
    if no_crit_damage:
        crit_dmg = regular_dmg
    else:
        crit_dmg = regular_dmg + damage + extra_dice
    out = choice([p_miss, p_regular_hit, p_crit], 0, regular_dmg, crit_dmg)
    attack_str = str(attack_roll).replace('+0', '')
    dmg_str = str(regular_dmg).replace('+0', '')
    ac_str = str(ac).replace('+0', '')
    match = re.search(r'\+[1-9][0-9]*$', ac_str)
    if match:
        temp1 = match.group()[1:]
        temp2 = ac_str[:match.start()]
        if temp1[0] == '-':
            temp = temp1[1:]
        ac_str = f'{temp1}+{temp2}'
    out.name = f'[{attack_str} vs AC {ac_str} dealing {dmg_str}'
    if crit_range != 20:
        out.name += f' (crit range {crit_range})'
    if no_crit_damage:
        out.name += " (crits deal regular damage)"
    elif extra_dice != 0:
        out.name += f', (enhanced crits deal extra {extra_dice})'
    out.name += ']'
    PRINT_COMPARISONS = True
    return out

crit = attack

def int_div_to_0(x, n):
    '''
    Internal function. Equivalent to x//n except negative values round towards 0.
    '''
    if x < 0:
        return int(-(-x//n))
    return int(x//n)

def is_number(x):
    '''Internal function. Returns True if x is a numeric type, False otherwise.'''
    return isinstance(x, (int, float, np.number))

def sample(d):
    '''
    Internal function for generating a sample from a distribution.
    d: An object of class die.
    Returns an integer.
    '''
    u = np.random.default_rng().random()
    return d.start + np.where(np.cumsum(d.arr) >= u)[0][0]

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

def stat_roll(): # Needs to have a different name than say 4d6dl() bc regex
    arr = np.array(
        [1, 4, 10, 21, 38, 62, 91, 122, 148, 167, 172, 160, 131, 94, 54, 21],
        dtype=np.float64
    )
    arr /= 1296
    return die(arr, 3, '4d6dl', True)

help_string = '''
Getting started: Try typing (2d6+3)/2

Available things:
 +, -, *, , / (, ):
   Does what you'd expect, eg 2d6-1d4+3, 3*1d6 (not the same as 3d6), 1d4*(2d6+2), 8d6/2.
   Division (or multiplying by a fraction) rounds towards 0.
   You can also do things like (1d4)d6, which represents rolling 1d4 then adding up that many d6.

 ^, **:
   Raise to a power, eg 1d3^2 is either 1, 4, or 9, 1d4**2 is either 1, 4, 9, or 16.
   Can only raise dice to positive integer powers.

 adv, dis:
   Gives the better/worse of 2 attempts, eg adv(1d20). 

 highest, lowest:
   Like adv() or dis() but with other numbers, eg highest(1d20, 3) is the best of 3 rolls.
   Syntax: highest(die, number of rolls), lowest(die, number of rolls)

 order:
   Like highest or lowest, but it can give middle values.
   Syntax: order(die, number of rolls, which roll to keep), where 1 means keep the lowest roll.
   Ex: order(4d6dl, 6, 2) gives your 2nd lowest roll from doing 4d6 drop lowest 6 times.

 min0, min1, min_val:
   min0, min1 mean results less than 0 or 1 are replaced with 0 or 1 respectively.
   min_val does the same but with an arbitrary minimum, so in min_val(1d6-3, 2),
   results less than 2 are replaced with 2.

 >, >=, <, <=, ==:
   3d4 > 2d6 prints the probability that 3d4 > 2d6, and the other symbols are similar.
   This returns a Bernoulli (binary) distribution, where P(1) is the probability of
   the expression being true, and P(0) is the probability of the expression being false.

 4d6dl or '4d6 drop lowest':
   This gives the distribution of 4d6 with the lowest die removed.

 attack or crit:
   This calculates everything related to a DnD 5e attack roll, including crits, expanded
   crit range, extra crit damage
   Basic syntax: crit(attack_bonus, enemy_ac, damage_dice, damage_bonus)
   Ex: attack(4, 16, 2d6, 3) for an attack with +4 to hit that deals 2d6+3 vs AC 16
   Ex: crit(4+1d4, 16, 2d6, 3, adv=True) for an attack with advantage, with
       +4+1d4 to hit, dealing 2d6+3 vs AC 16
   Enter "help attack" for more advanced usage of this function.

 @:
   More advanced version of (1d4)d6. If an attack hits 1d6 times, and each hit
   deals 2d10+1d4, then you can write that as 1d6 @ (2d10+1d4).

 choice:
   Function for if/else statements. Multiple possible syntaxes.
   Syntax 1: choice(condition, if_false, if_true)
   Syntax 2: choice(probability_of_true, if_false, if_true)
   Syntax 3: choice(distribution, value1, value2, ...)
   Enter "help choice" for other ways to use choice'''

choice_help = '''
Syntax 1: choice(condition, if_false, if_true)
Gives the distribution of (if_true if condition is true, otherwise if_false)
Ex: choice(1d20+3>=15, 8d6, 8d6/2) is the distribution of the following process:
1. roll 1d20+3
2. if the result is at least 15, return 8d6/2
3. otherwise return 8d6
This models a +3 dex save against fireball with DC 15

Syntax 2: choice(probability_of_true, if_false, if_true)
This gives (if_true with probability probability_of_true, otherwise if_false).
Ex: choice(.6, 1d5, 0) is the distribution of
(0 with probability .6 and 1d5 with probability .4)

Syntax 3: choice(dist, value1, value2, ...)
This gives the distribution of
(value1 if distr gives its first value, value2 if distr gives its second value, etc)
distr can be a die expression or an array.
Ex: choice(2d2, 1d6, 1d8, 1d10) gives the distribution of rolling 2d2 and
returning 1d6 if you got 2, 1d8 if you got 3, and 1d10 if you got 4.
Ex: choice([.2, .5, .3], 1d4, 1d6, 1d8) means 
1d4 with probability .2, 1d6 with probability .5, and 1d8 with probability .3'''

attack_help = '''
Syntax:
attack(bonus, ac, damage, damage_bonus, [extra_dice, crit_range, adv, no_crit_damage])
The arguments in [...] are optional but must be supplied by name, after all
of the mandatory arguments, ie:
attack(4, 16, 1d6, 3, crit_range=19, adv=True).

Arguments:
bonus: The attack bonus, ie 4 or 4+1d4
ac: The target AC, ie 16 or 14+1d8
damage: The damage dice, ie 2d6
damage_bonus: The fixed number always added to the damage, ie 3
extra_dice: Additional dice added to crits, for features like brutal critical
crit_range: The attack crits on a d20 roll of this number or higher
adv: 1 or True for advantage, 2 for double advantage, 3 for triple advantage, etc.
     -1 for disadvantage, -2 for double disadvantage, etc
no_crit_damage: If True, all bonus damage for crits is blocked. This means that
                crits auto-hit but deal regular attack damage.'''

if __name__ == '__main__':
    if len(sys.argv) > 1:
        process_input(' '.join(sys.argv[1:]))
        exit()
    print(help_string)
    while True:
        print('\nEnter q to quit, h or help for help')
        text = input('>>')
        if text.lower() == 'q' or text.lower() == 'exit':
            exit()
        if text.lower() == 'h' or text.lower() == 'help':
            print(help_string)
            continue
        if text.lower() == 'help choice' or text.lower() == '"help choice"':
            print(choice_help)
            continue
        if text.lower() == 'help attack' or text.lower() == '"help attack"':
            print(attack_help)
            continue
        if len(text) > 0:
            try:
                process_input(text)
            except NameError as e:
                print('Not a valid input.')
                traceback.print_exc()
            except Exception as e:
                print('Error encountered, aborting input.')
                traceback.print_exc()
