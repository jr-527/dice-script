# This code taken (and slightly modified)
# from here: https://gist.github.com/vyznev/8f5e62c91ce4d8ca7841974c87271e2f
# and from https://math.stackexchange.com/questions/3765873/
from functools import reduce
from collections import defaultdict
from numpy.polynomial import polynomial as p

dice_cache = {}
n_iterations = 0
def dice_roll(die, count = 1, select = None):
    """Generate all possible results of rolling `die` `count` times, sorting
    the results (according to the order of the sides on the die) and selecting
    the first `select` elements of it.
    The yielded results are tuples of the form `(roll, prob)`, where `roll` is a
    sorted tuple of `select` values and `prob` is the probability of the result.
    The first argument can be either a custom die, i.e. a tuple of `(side, prob)`
    pairs, where `prob` is the probability of rolling `side` on the die, or just
    a simple integer, which will be passed to `make_simple_die`.
    Keyword arguments:
    die -- a custom die or an integer
    count -- the number of dice to roll (default 1)
    select -- the number of results to select (set equal to count if omitted)
    """
    # cannot select more dice than there are in the pool
    if select is None or select > count:
        select = count
    # for convienience, allow simple dice to be given as plain numbers
    die = make_simple_die(die) if isinstance(die, int) else tuple(die)
    cache_me = False
    if (die, count, select) in dice_cache:
        thing = dice_cache[(die, count, select)]
        for x in thing:
            yield x
        return
    else:
        cache_me = True
        out = []
    if len(die) == 1:
        # base case: a one-sided die has only one possible result
        if cache_me:
            out.append(((die[0][0],) * select, die[0][1]**count))
        yield ((die[0][0],) * select, die[0][1]**count)
    elif len(die) > 1:
        # split off the first side of the die, normalize the rest
        side, p_side = die[0]
        rest = tuple((side, prob / (1-p_side)) for side, prob in die[1:])
        p_sum = 0 # probability of rolling this side less than select times
        for i in range(0, select):
            # probability of rolling this side exactly i times
            p_i = binomial(count, i) * p_side**i * (1-p_side)**(count-i)
            p_sum += p_i
            # recursively generate combinations
            if (rest, count-i, select-i) in dice_cache:
                thing = dice_cache[(rest, count-i, select-i)]
            else:
                thing = list(dice_roll(rest, count-i, select-i))
                dice_cache[(rest, count-i, select-i)] = thing
            # for roll, p_roll in dice_roll(rest, count-i, select-i):
            for roll, p_roll in thing:
                if cache_me:
                    out.append(((side,) * i + roll, p_i * p_roll))
                yield ((side,) * i + roll, p_i * p_roll)
        # final case: all selected dice (and possibly more) roll this side
        if cache_me:
            out.append(((side,) * select, 1-p_sum))
        yield ((side,) * select, 1-p_sum)
        if cache_me:
            dice_cache[(die, count, select)] = out


_factorials = [1]
def binomial(n, k):
    """Helper function to efficiently compute the binomial coefficient."""
    while len(_factorials) <= n:
        _factorials.append(_factorials[-1] * len(_factorials))
    return _factorials[n] / _factorials[k] / _factorials[n-k]

def make_simple_die(n):
    """Generate a simple n-sided die with sides listed in decreasing order."""
    return tuple((i, 1.0/n) for i in range(n, 0, -1))

def explode(die, count=2):
    """Make an "exploding die" where the first (=highest) side is rerolled up to
    count times.
    """
    die = make_simple_die(die) if isinstance(die, int) else tuple(die)
    exploded = die
    for i in range(count):
        top, p_top = exploded[0]
        exploded = tuple((side + top, prob * p_top) for side, prob in die) + exploded[1:]
    return exploded

def sum_roll(die, count = 1, select = None, ascending=False):
    """Convenience function to sum the results of `dice_roll()`. Takes the same
    parameters as `dice_roll()`, returns a list of `(sum, prob)` pairs sorted in
    descending order by sum (and thus suitable for use as a new custom die). The
    optional parameter `ascending=True` can be used to change the sort order.
    """
    summary = defaultdict(float)
    for roll, prob in dice_roll(die, count, select):
        summary[sum(roll)] += prob
    return tuple(sorted(summary.items(), reverse = not ascending))

def generating_function(k, d, n):
    return p.polypow([0] * k + [1] * (d - k + 1), n)

def drop_one_die(n, d):
    tmp = [generating_function(k, d, n) for k in range(1, d + 2)]
    differences = ((tmp[i] - tmp[i + 1])[i + 1:] for i in range(d))
    return reduce(p.polyadd, differences)