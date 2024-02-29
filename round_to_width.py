_NEG_OF = 'negative overflow'
_OF = 'overflow'
_OK = 'okay'

def _round_to_1(x):
    if x < -0.5: return ('0', _NEG_OF)
    if x <= 0.5: return ('0', _OK)
    if x >= 9.5: return ('9', _OF)
    return (str(int(round(x))), _OK)

def _round_to_2(x, leading_zero=False):
    if x <= -9.5: return ('-9', _NEG_OF)
    if x < -0.5: return (str(int(round(x))), _OK)
    if x < 0: return ('-0', _OK)
    if not leading_zero and x < 1:
        if 0.95 <= x: return ('1', _OK)
        if 0.05 >= x: return ('0', _OK)
        return (str(int(round(10*x))/10.0)[1:3], _OK)
    if x >= 99.5: return ('99', _OF)
    return (str(int(round(x))), _OK)

def _round_to_3(x, leading_zero=False):
    if x < 0:
        out = _round_to_2(-x, leading_zero)
        if out[1] == _OF: return ('-99', _NEG_OF)
        return ('-' + out[0], _OK)
    if not leading_zero and x < 1:
        if 0.995 <= x: return ('1', _OK)
        if 0.005 >= x: return ('0', _OK)
        return (str(round(x,2))[1:4], _OK)
    if x < 1:
        return ('0' + _round_to_2(x, False)[0], _OK)
    if x < 9.95: return (str(round(x,1)), _OK)
    if x < 999.5: return (str(int(round(x))), _OK)
    if 999.5 <= x and x < 9.5e9:
        out = format(x, '1e')
        first_digit = int(round(float(out[:3])))
        last_digit = int(out[-1])
        if first_digit == 10:
            last_digit += 1
            first_digit = 1
        return (str(first_digit) + 'e' + str(last_digit), _OK)
    return ('9e9', _OF)

def _round_to_n(x, n, leading_zero=False):
    if n >= 25:
        return (str(x), _OK)
    if n == 3:
        return _round_to_3(x, leading_zero)
    if x < 0:
        out, status = _round_to_n(-x, n-1, leading_zero)
        status = status if status == _OK else _NEG_OF
        return ('-' + out, status)
    if x <= float('0.5e-' + '9'*(n-3)):
        return ('0', _OK)
    if x < float('1.5e-' + '9'*(n-3)):
        return ('1e-' + '9'*(n-3), _OK)
    if x < .001:
        mantissa, exponent = format(x, '.18e').split('e')
        exponent = 'e' + str(int(exponent))
        mantissa = str(round(float(mantissa), max(0,n-len(exponent)-2)))
        mantissa = mantissa[:n-len(exponent)]
        if mantissa[-1] == '.':
            mantissa = mantissa[:-1]
        return (mantissa + exponent, _OK)
    threshold_str = '9'*n + '5'
    for i in range(n+1):
        threshold = float(threshold_str[:i] + '.' + threshold_str[i:])
        if x < threshold:
            out = ''
            if i == 0:
                if not leading_zero:
                    out = str(round(x,n-i-1))[1:]
                else:
                    out = str(round(x,n-i-2))
            if i == n:
                out = str(int(round(x)))[:n]
            else:
                out = str(round(x,n-i-1))[:n]
                if out[-1] == '.':
                    #out = str(round(x,n-i-1))[:n-1]
                    out = out[:-1]
            out = out[:n]
            if out[-1] == '.':
                out = out[:-1]
            return (out, _OK)
    if x < float('9.5e' + '9'*(n-2)):
        mantissa, exponent = format(x, '.17e').split('e')
        exponent = 'e' + str(int(exponent))
        mantissa = str(round(float(mantissa), max(0,n-len(exponent)-2)))
        mantissa = mantissa[:n-len(exponent)]
        if mantissa[-1] == '.':
            mantissa = mantissa[:-1]
        return (mantissa + exponent, _OK)
    return (float('9e' + '9'*(n-2)), _OF)

def round_to_width(x, width=8, align='left', overflow='saturate', leading_zero=False, underflow='zero'):
    '''
    Returns a string of length width representing a number as close to x as
    possible (prefers '1.2e9' over '123e7' because it's easier to understand)
    x: The number to round
    width: The string's width
    align: If 'left', align the string on the left, ie '3.14   '.
           Otherwise align the string on the right, ie '   3.14'
    overflow: Either 'saturate', 'exception', 'inf', 'nan', or 'word'
             Describes what to do if the number is too big to represent in the
             desired number of characters.
        'saturate': Give the closest possible number, so
                    round_to_width(1e20, 3) gives '9e9'
        'exception': Raise an exception.
        'inf': Return 'inf' or '-inf' (replaced by 'exception' if width < 4)
        'nan': Return 'NaN' or '-NaN' (replaced by 'exception' width < 4)
        'word': Returns text that says something along the lines of 'overflow'
                that won't parse as a number.
    underflow: Either 'zero', 'saturate', 'exception', or 'word'
               Describes what to do if the number is too close to zero to
               represent in the desired number of characters.
        'zero': Return '0'
        'saturate': Give the smallest possible number of the correct sign, so
                    round_to_width(1e-99, 4) gives '1e-9'.
                    Replaced by 'exception' if width < 2.
        'exception': Raise an exception.
        'word': Returns text that says something along the lines of 'eps'
                that won't parse as (part of) a number (so no 'e').
                Replaced by 'exception' if width < 3.
    leading_zero: If True, 0.5 will be formatted as '0.5', otherwise '.5'
    '''
    width = int(width)
    if width == 0:
        # returning '' is a silent failure, which we don't want.
        raise ValueError('width must be > 0')
    overflow = overflow.lower()
    underflow = underflow.lower()
    if underflow in ('raise', 'except'):
        underflow = 'exception'
    if (underflow == 'word' and width < 3) or (underflow == 'word' and width < 2):
        underflow = 'exception'
    if overflow in ('raise', 'except') or (width < 4 and overflow in ('inf', 'nan')):
        overflow = 'exception'
    if overflow not in ('exception', 'saturate', 'inf', 'nan', 'word'):
        raise ValueError("overflow must be one of 'exception', 'saturate', 'inf', 'nan', 'word'")
    if underflow not in ('zero', 'saturate', 'exception', 'word'):
        raise ValueError("underflow must be one of 'zero', 'saturate', 'exception', 'word'")
    out, status = '', ''
    if width == 1:
        out, status = _round_to_1(x)
    elif width == 2:
        out, status = _round_to_2(x, leading_zero)
    else:
        out, status = _round_to_n(x, width, leading_zero)
    if out == '0' and x != 0: # this branch always has status == _OK
        # 'zero' -> do nothing
        if underflow == 'exception':
            raise ValueError('Underflow')
        if underflow == 'saturate':
            # the smallest floating point number is around 1e-308,
            # so we don't have to worry about 1e-999.
            temp_arr = [('1', '1'), ('.1', '1'), ('.01', '0.1'),
                        ('1e-9', '1e-9'), ('1e-99', '1e-99')]
            if x > 0: out = temp_arr[width-1][leading_zero]
            else: out = '-' + temp_arr[width-2][leading_zero]
        elif underflow == 'word':
            out = 'eps'
            if x < 0:
                out = '-ep' if width == 3 else '-eps'
    fmt = '<' if align=='left' else '>'
    out = format(out, fmt+str(int(width)))
    if status != _OK:
        # saturate -> do nothing
        if overflow == 'exception':
            raise ValueError(status)
        if overflow == 'inf':
            out = 'inf' if status == _OF else '-inf'
        elif overflow == 'nan':
            status = 'NaN' if status == _OF else '-NaN'
        elif overflow == 'word':
            if width < 5:
                state = 1 if status == _OF else 0
                return [('N', 'P'), ('-N', 'OF'), ('-OF', 'OVF'), ('-OVF', 'OVER')][width-1][state]
            out = 'OVERFLOW' if status == _OF else '-OVERFLOW'
            out = out[:min(len(out), width)]
    return out
