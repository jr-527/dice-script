# This code modified from
# https://github.com/HighDiceRoller/icepool/blob/main/papers/icepool_preprint.pdf

# The following license applies to this file only, I think:
# MIT License

# Copyright © 2021-2024, Albert Julius Liu. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import math
import collections
from functools import cache
import numpy as np

@cache
def _solve(faces: int, n: int, keep: int) -> dict:
    '''
    Internal function, implements the logic in drop_die but returns
    a dictionary rather than the usual (start, pmf)
    '''
    # outcome = faces
    # faces -= 1
    if faces == 1:
        state = faces * min(n, keep)
        return {state : 1}
    result = collections.defaultdict(int)
    for k in range(n + 1):
        tail = _solve(faces-1, n - k, keep=keep-min(keep, k))
        for state, weight in tail.items():
            state = state + faces * min(keep, k)
            weight *= math.comb(n, k)
            result[state] += weight
    return result

# not to be confused with "drop dead"
def drop_die(faces: int, n: int, keep: int) -> tuple[int, np.ndarray]:
    '''
    Calculates the PMF of rolling n die, where each die has faces sides,
    then throwing out all but the top "keep" die, and returning the sum
    of the remaining die.
    faces: An int, the number of faces on each die
    n: An int, the number of die
    keep: An int, we throw out the lowest die until there are this many left

    Returns: (start, arr), where start is an int, arr is a numpy array,
    such that arr[x-start] is the probability of getting x
    '''
    out = []
    for _, v in _solve(faces, n, keep).items():
        out.append(v)
    out = np.array(out)
    return keep, out / np.sum(out)
