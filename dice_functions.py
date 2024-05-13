'''User-facing functions'''
from numbers import Real
from typing import Callable
import numpy as np
from die import die, ndm, bin_coeff, PRINT_COMPARISONS
from drop import drop_die
import re

def min0(d: die) -> die:
    '''
    Transforms d so that values less than 0 are increased to 0.
    d: A die class object
    Returns a new die class object.
    '''
    return min_val(d, 0)

def min1(d: die) -> die:
    '''
    Transforms d so that values less than 1 are increased to 1.
    d: A die class object
    Returns a new die class object.
    '''
    return min_val(d, 1)

def min_val(d: die, m: int = 1):
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

def mean(d: die) -> float:
    '''Internal function, returns the mean of a die class object.'''
    x = np.arange(d.start, d.start+len(d.arr))
    out = float(np.sum(x * d.arr))
    if abs(out) < 2**(-53): # values below this are likely rounding artifacts
        out = 0.0 # so it's safer to just round to 0
    return out

def var(d: die) -> float:
    '''Internal function, returns the variance of a die class object.'''
    x = np.arange(d.start, d.start+len(d.arr))
    mu = mean(d)
    return max(float(np.sum(d.arr * (x-mu)**2)), 0.0)

def sd(d: die) -> float:
    '''Internal function, returns the standard deviation of a die class object.'''
    return float(np.sqrt(var(d)))

def order_stat(x: die, num: int = 1, pos: int = 1) -> die:
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

def highest(x: die, n: int = 2) -> die:
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

def adv(x: die) -> die:
    '''
    Returns the distribution of sampling twice from x and keeping the greater sample.
    x: A die class object
    Returns a new die class object.
    '''
    return highest(x, 2)

advantage = adv

def lowest(x: die, n: int = 2) -> die:
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

def disadv(x: die) -> die:
    '''
    Returns the distribution of sampling twice from x and keeping the lower sample.
    x: An object of class die.
    '''
    return lowest(x, 2)

dis = disadv
disadvantage = disadv

def choice(condition, *args) -> die:
    '''
    Returns the distribution of choosing an argument based on a sample from condition.
    condition: A probability distribution. Can be a list of non-negative numbers that
        sums to 1 or a die class. Note that "boolean" die expressions, eg 1d20 > 3,
        are Bernoulli RVs, so distributions with length 2.
    *args: Distributions to simulate samples from based on the result of condition. Number of
        distributions must be same as length of condition. Can be integers or die class objects.
    Ex:
    choice([0.4, 0.4, 0.2], 1d4, 1d6, 2d4)
    Returns the distribution that's a 40% chance of 1d4, 40% of 1d6, and 20% chance of 2d4

    choice(adv(1d20+4) >= 15, 8d6/2, 8d6)
    Returns the distribution that's 8d6/2 if adv(1d20+4) < 15, 8d6 otherwise

    choice(1d4, 1, 3, 3, 7)
    Returns the distribution that's 1 with probability 25%, 3 with prob. 50%, 7 with prob. 25%.
    '''
    n = len(args)
    probs = []
    if not isinstance(condition, die):
        if isinstance(condition, Real):
            condition = float(condition)
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
        if isinstance(thing, Real):
            thing = float(thing)
            lb = min(lb, round(thing))
            rb = max(rb, round(thing))
        elif isinstance(thing, die):
            lb = min(lb, thing.start)
            rb = max(rb, thing.start+len(thing.arr))
        else:
            raise TypeError('Invalid argument type for choice()')
    if lb > rb or not isinstance(lb, int) or not isinstance(rb, int):
        raise ValueError('Error in choice()')
    out = np.zeros(rb-lb+1)
    for thing, p in zip(args, probs):
        if isinstance(thing, Real):
            out[round(float(thing))-lb] += p
        else:
            padded = np.hstack([
                [0.0]*(thing.start-lb),
                thing.arr,
                [0.0]*(rb-(thing.start+len(thing.arr)-1))
            ])
            out += padded * p
    if not np.isclose(sum(out), 1):
        raise ValueError('Error in choice(): result is not a PMF')
    out_string = f'choice({condition}'
    for arg in args:
        out_string += f', {arg}'
    out_string += ')'
    return die(out, lb, out_string, True)

def attack(bonus: int|die, ac: int|die, damage:int|die, damage_bonus:int|die=0,
           *, extra_dice:int|die|None=None, crit_range:int=20, adv:int=0,
           no_crit_damage:bool=False) -> die:
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
    PRINT_COMPARISONS[0] = False
    d20 = die(ndm(1,20), 1, '1d20', True)
    pos_val = 1 if (adv <= 0) else abs(adv)+1
    d20_roll = order_stat(d20, abs(adv)+1, pos_val)
    attack_roll = d20_roll + bonus
    attack_roll.basicName = True
    p_crit = (d20_roll >= crit_range)[1]
    # We want P(miss because natural 1)
    # =P(attack roll >= ac|nat 1)*P(nat 1)
    blocked_hit = (bonus+1) >= ac
    if isinstance(blocked_hit, die):
        if isinstance(ac, die):
            ac.basicName = True
        blocked_hit = blocked_hit[1]
    p_relevant_nat1 = blocked_hit * (d20_roll == 1)[1]
    # the previous line is either a number or a die object
    # if it's a number, we're done. If it's a die object,
    # we need the following line to convert it into a number
    # if isinstance(p_relevant_nat1, (list, np.ndarray, die)):
    #     p_relevant_nat1 = p_relevant_nat1[1]
    p_regular_hit = (attack_roll >= ac)[1] - p_crit - p_relevant_nat1
    # In some cases a regular hit is impossible, ie AC 100
    p_regular_hit = max(0.0, p_regular_hit)
    p_miss = 1.0-p_regular_hit-p_crit
    regular_dmg = damage + damage_bonus
    if isinstance(regular_dmg, die):
        regular_dmg.basicName = True
    crit_dmg = None
    if no_crit_damage:
        crit_dmg = regular_dmg
    else:
        if extra_dice is None:
            crit_dmg = regular_dmg + damage
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
        ac_str = f'{temp1}+{temp2}'
    out.name = f'[{attack_str} vs AC {ac_str} dealing {dmg_str}'
    if crit_range != 20:
        out.name += f' (crit range {crit_range})'
    if no_crit_damage:
        out.name += " (crits don't deal extra damage)"
    elif extra_dice is not None and not no_crit_damage:
        out.name += f' (enhanced crits deal extra {extra_dice})'
    out.name += ']'
    PRINT_COMPARISONS[0] = True
    return out

crit = attack

def check(bonus: int|die, dc: int|die, adv: int=0, *, succeed: die|None = None,
          fail: die|None = None) -> die:
    '''
    Returns the distribution of passing/failing a check, with the specified bonus,
    against the specified DC, possibly with advantage/disadvantage. Similar to
    the attack function.
    If succeed and fail are specified, then this returns the distribution of
    attempting the check, rolling succeed if it passes and fail if it fails.
    '''
    global PRINT_COMPARISONS
    PRINT_COMPARISONS[0] = False
    d20 = die(ndm(1,20), 1, '1d20', True)
    pos_val = 1 if (adv <= 0) else abs(adv)+1
    d20_roll = order_stat(d20, abs(adv)+1, pos_val)
    blocked_check = (bonus+1) >= dc
    if isinstance(dc, die):
        dc.basicName = True
    if isinstance(blocked_check, die):
        blocked_check = blocked_check[1]
    p_relevant_nat1 = blocked_check * (d20_roll == 1)[1]
    if isinstance(p_relevant_nat1, die):
        p_relevant_nat1 = p_relevant_nat1[1]
    roll_with_bonus = d20_roll + bonus
    roll_with_bonus.basicName = True
    p_succeed = (roll_with_bonus >= dc)[1] - p_relevant_nat1
    PRINT_COMPARISONS[0] = True
    out = die([1-p_succeed, p_succeed], 0, f'[{roll_with_bonus} vs DC {dc}]', True)
    if succeed is not None and fail is not None:
        out = choice(out, succeed, fail)
        out.name = f'[{roll_with_bonus} vs DC {dc}, success: {succeed}, fail: {fail}]'
        out.basicName = True
    return out

save = check

def sample(d: die) -> int:
    '''
    Internal function for generating a sample from a distribution.
    d: An object of class die.
    Returns an integer.
    '''
    u = np.random.default_rng().random()
    return d.start + np.where(np.cumsum(d.arr) >= u)[0][0]

def multiple_inequality(*args: float|die) -> die:
    '''
    Internal function for evaluating chained inequalities.
    Returns a number.
    The first, third, fifth, ... arguments should be numbers or die objects
    The second, fourth, sixth, ... arguments should be one of '>', '<', '>=',
    '<=', '==', '!='.
    ex: P(A <= B) == multiple_inequalities(A, '<=', B)
        P(A < B == C) == multiple_inequalities(A, '<', B, '==', C)
        P(A < B < C < D) == multiple_inequalities(A, '<', B, '<', C, '<', D)
    '''
    # In Python, w < x <= y > z
    # is equivalent to
    # (w < x) and (x <= y) and (y > z)
    # In pseudocode, for "w < x <= y > z == ...", this function does:
    # out = 0
    # for each possible value i of w:
    #     for each possible value j of x:
    #         for each possible value k of y:
    #             for ...
    #                 if i < j <= k > l == ...:
    #                     out += P(w==i) * P(x==j) * P(y==k) * ...
    # return out
    relations = {
        '>':np.greater, '<':np.less, '>=':np.greater_equal,
        '<=':np.less_equal, '==':np.equal, '!=':np.not_equal
    }
    # the types can't be wrong for the next few lines if the function is
    # called correctly
    arrs: list[np.ndarray] = [
        np.array([1.0]) if isinstance(x, Real)
        else x.arr for x in args[::2]] # type: ignore
    starts: list[float] = [x if isinstance(x, Real) else x.start for x in args[::2]] # type: ignore
    ops: list[Callable[[list[float], list[float]], list[np.bool_]]] = [
        relations[x] for x in args[1::2]] # type: ignore
    # We use einsum because np.outer doesn't play nicely with 3+ arrays.
    # It performs the P(w==i)*P(x==j)*P(y==k)*... part of the pseudocode,
    # but vectorized.
    es_str = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'[:len(args)]
    prod = np.einsum(es_str, *arrs)
    # np.indices lets us keep track of i, j, k, ...
    indices = np.indices(prod.shape)
    # bools[i,j,k,...] = (i+w_start < j+x_start <= k+y_start > ...)
    # ops[0]( ... -starts[0])) does the i < j check, but vectorized
    # ops[1]( ... -starts[1])) does the j <= k check, but vectorized
    # etc
    # We use np.all(..., 0) to coalesce all that
    # (0 corresponds to the "for i in range(len(ops))" axis)
    bools = np.all(
        [
            ops[i](
                indices[i],
                indices[i+1]+(starts[i+1]-starts[i]))
            for i in range(len(ops))
        ],
        0)
    # This does the "out +=" part of the pseudocode, but vectorized
    p = np.sum(bools * prod)
    return die(np.array([1-p, p]), 0, ''.join((str(x) for x in args)), True, True)

def drop(count: int, faces: int, mode: str, n: int = 1) -> die:
    '''
    Calculates the PMF of rolling count die, where each die has faces sides.
    If mode is "keep highest", we keep the highest n die, throw out
    the rest, then take the sum. Any combination of "keep"/"drop" and
    "highest"/"lowest" works for mode.
    count: int, the number of die
    faces: int, the number of faces on each die
    mode: string. Any combination of "keep"/"drop" and "highest"/"lowest", ie
        "keep highest" is valid. You can also abbreviate words to their first
        letter, ie "kh" for "keep highest" or "dl" for "drop lowest"
    n: int, the number of die either kept or discarded, depending on mode

    Returns a new die class object.
    '''
    select = n
    ascending = False
    if 'k' in mode:
        if 'l' in mode:
            ascending = True
    else:
        select = count-n
        if 'h' in mode: # dh
            ascending = True
    start, x = None, None
    if select == count:
        x = ndm(n=count, m=faces)
        start = count*faces
    else:
        start, x = drop_die(faces, count, select)
        x = x / np.sum(x)
        if ascending:
            x = np.flip(x)
    name, basic = '', False
    if ' ' in mode:
        name = f'{count}d{faces} {mode} {n}'
    else:
        name = f'{count}d{faces}{mode}{n}'
        basic = True
    return die(x, start, name, basic)
