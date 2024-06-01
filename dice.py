#!/usr/bin/env python3
'''
Implements dice and mathematical operations on dice.
This module can be run by executing the main() function, which activates REPL
functionality. It also provides an API of sorts, in the form of the "handle"
function, which allows you to do things like
    d, f = handle('3d7/2 > 5')
    f.show()
to get a nice plot.
'''
from numbers import Real
from die import die, ndm
import numpy as np
from dice_functions import (min0, min1, min_val, mean, var, sd, order_stat,
    order, highest, adv, advantage, lowest, disadv, dis, disadvantage, choice,
    attack, crit, check, save, sample, multiple_inequality, drop)
from round_to_width import round_to_width as round_w
import dice_strings
import sys
# On my older laptop, this file's startup time was about .5 seconds, most of
# which was spent in the line "import matplotlib.pyplot as plt"
# To speed this up, we perform the import asynchronously, as plt isn't needed
# until after the user types something in and presses enter, and this way
# the user can type things while matplotlib is being imported.
plt = None
import threading
import re
import warnings
import traceback
import ast
warnings.filterwarnings('ignore', 'elementwise comparison failed')

plt_initialized = False
def import_plt():
    global plt
    global plt_initialized
    import matplotlib.pyplot as plt
    plt_initialized = True
import_thread = threading.Thread(target=import_plt, name='import matplotlib')
import_thread.start()

__all__ = ['main', 'handle', 'plot', 'process_input', 'd', 'min0', 'min1',
           'min_val', 'mean', 'var', 'sd', 'order_stat', 'order', 'highest',
           'adv', 'advantage', 'lowest', 'disadv', 'dis', 'disadvantage',
           'choice', 'attack', 'crit', 'check', 'save', 'sample',
           'multiple_inequality', 'drop']

safe_functions = set((
    'd',
    'print',
    'help',
    'min0',
    'min1',
    'min_val',
    'order_stat',
    'order',
    'highest',
    'adv',
    'advantage',
    'lowest',
    'disadv',
    'dis',
    'disadvantage',
    'choice',
    'attack',
    'crit',
    'check',
    'save',
    'multiple_inequality',
    'drop',
    'die',
    'ndm',
    'reroll',
    '_comparison',
    '_equals',
))

safe_nodes = set((
    ast.Expression,
    ast.Module,
    ast.Constant,
    ast.Expr,
    ast.Num,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Attribute,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add, ast.UAdd, ast.Sub, ast.USub,
    ast.Mult, ast.MatMult,
    ast.Div, ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE,
    ast.Gt, ast.GtE,
    ast.BitOr
))

compare_ops = {
    ast.Eq:ast.Constant('=='),
    ast.NotEq:ast.Constant('!='),
    ast.Lt:ast.Constant('<'),
    ast.LtE:ast.Constant('<='),
    ast.Gt:ast.Constant('>'),
    ast.GtE:ast.Constant('>='),
}

class _MultipleIneq(ast.NodeTransformer):
    '''
    Class to convert expressions like a<b<c into the function call
    multiple_inequality(a, '<', b, '<', c)
    '''
    def visit_Compare(self, node):
        if len(node.comparators) == 1:
            return node
        values = [node.left]
        operators = [compare_ops[x.__class__] for x in node.ops] # type: ignore
        # this comprehension interleaves operators and node.comparators
        values += [x for y in zip(operators, node.comparators) for x in y]
        function = ast.Name(
            id='multiple_inequality',
            ctx=ast.Load()
        )
        return ast.copy_location(ast.Call(function, values, []), node)

def whitelist_eval(string: str) -> die:
    '''
    Checks that the AST of string only contains nodes and function calls
    included in the whitelist. If so, evaluates string, otherwise raises
    an exception. Also converts sub-expressions like a < (b+c) < d into
    multiple_inequality(a, '<', (b+c), '<', d)
    '''
    tree = ast.parse(string, mode='eval')
    tree = _MultipleIneq().visit(tree)
    for node in ast.walk(tree):
        if type(node) not in safe_nodes:
            raise Exception(f'{type(node)} is not a whitelisted operation')
        elif isinstance(node, ast.Call):
            func = ''
            if isinstance(node.func, ast.Attribute):
                func = node.func.attr
            elif isinstance(node.func, ast.Name):
                func = node.func.id
            else:
                raise Exception(f'{type(node.func)} is not a whitelisted call type')
            if func not in safe_functions:
                raise Exception(f'{func} is not a whitelisted function')
    ast.fix_missing_locations(tree)
    return eval(compile(tree, filename='<whitelisted-ast>', mode='eval'))

def d(x: int, y: int):
    '''Wrapper function for die(ndm(x,y),..)'''
    return die(ndm(x, y), x, f'{x}d{y}', True)

def process_input(text: str) -> die|float|None:
    '''
    Internal function. Processes then calls eval on text.
    '''
    new_text = re.sub(' = ', ' == ', text)
    new_text = re.sub(r'[)]d', r')@1d', new_text)
    drop_regexp = r'([1-9][0-9]*)d([1-9][0-9]*)\s*(kh|dh|kl|dl|keep lowest|' + \
        r'keep highest|drop lowest|drop highest)\s*([1-9][0-9]*)?'
    new_text = re.sub(drop_regexp, r'drop(\1, \2, "\3", \4)', new_text)
    reroll_regexp = r'([1-9][0-9]*)d([1-9][0-9]*)(ro?)([<>]?[1-9][0-9]*|\[[^\]]*\])'
    new_text = re.sub(reroll_regexp, r'\1@(1d\2).reroll("\3", "\4")', new_text)
    reroll_regexp2 = r'\)(ro?)([<>]?[1-9][0-9]*|\[[^\]]*\])'
    new_text = re.sub(reroll_regexp2, r').reroll("\1", "\2")', new_text)
    new_text = new_text.replace('"<', '"l')
    new_text = new_text.replace('">', '"g')
    new_text = re.sub(r'\^', '**', new_text)
    new_text = re.sub(r'([1-9][0-9]*)d([1-9][0-9]*)', r'd(\1,\2)', new_text)
    try:
        x = whitelist_eval(new_text)
    except Exception as e:
        e.args = (e.args[0] + f'\nnew_text: {new_text}',)
        raise e.with_traceback(e.__traceback__)
    return x

def plot(d: die, name: str, print_stuff=True) -> 'matplotlib.pyplot.Figure|None': # type: ignore
    '''
    Internal function. Plots a distribution
    d: A number or die class object
    name (optional): A custom name for the plot.
    '''
    global plt_initialized, plt
    # If matplotlib isn't imported yet, we wait
    if not plt_initialized:
        import_thread.join()
        plt_initialized = True
    assert plt is not None
    if d.isProbability:
        p = mean(d)
        if print_stuff:
            if 0 < abs(p) < 1e-15:
                print('Possible rounding errors.')
            print('Probability:',
                f"{round_w(p,15,'left',leading_zero=True).strip()}",
                'standard deviation:',
                f"{round_w(sd(d),15,'left',leading_zero=True).strip()}")
            s = int(sample(d))
            s = 'Yes' if s else 'No'
            print(f'Random sample from distribution:', s)
            if d.arr[0] == 0 or d.arr[0] == 1:
                print('Nothing to plot.')
                return
        # print('Plotting in other window. That window must be closed to continue.')
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(name) # type: ignore (linter is wrong)
        plt.title('Distribution of ' + name)
        ax.bar(0, d.arr[0], 0.4, bottom=d.arr[1], label='No', color='tab:red')
        ax.bar(0, d.arr[1], 0.4, bottom=0, label='Yes', color='tab:blue')
        ax.set_xlim(-1,1)
        ax.get_xaxis().set_ticks([])
        ax.legend()
        return fig
    if print_stuff:
        print('Mean:',
            f"{round_w(mean(d),15,'left',leading_zero=True).strip()}",
            'standard deviation:',
            f"{round_w(sd(d),15,'left',leading_zero=True).strip()}")
        print(f'Random sample from distribution: {sample(d)}')
    # print('Plotting in other window. That window must be closed to continue.')
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(name) # type: ignore
    plt.title('Distribution of ' + name)
    y = d.arr
    cumulative = np.cumsum(y)
    # For larger arrays, we plot differently to reduce clutter.
    # For very large arrays, we split the array into buckets and plot the max of
    # each bucket, to improve speed.
    threshold_small_heads = 100
    threshold_no_heads = 1000
    threshold_continuous = 10000
    if len(y) >= threshold_continuous:
        col_size = int(len(y)/(threshold_continuous/2))
        # reshape y to (col_size, ?)
        new_cols = int(np.ceil(len(y)/col_size))
        y = np.max( # resample by taking the max of every ? elements
            np.pad(y, (0,new_cols*col_size-len(y))).reshape(new_cols, col_size),
            1
        )
        new_cdf = np.pad(cumulative, (0,new_cols*col_size-len(cumulative)),
            constant_values=cumulative[-1]).reshape(new_cols, col_size)
        cumulative = new_cdf[:,-1]
        x = np.arange(len(y))*col_size - d.start
        ax.plot(x, y, label='Probability', color='tab:blue')
        ax.fill_between(x,y, color='tab:blue', alpha=.5)
    else:
        x = range(d.start, d.start+len(y))
        if len(y) < threshold_small_heads:
            ax.stem(x, y, label='Probability', basefmt='')
        elif len(y) < threshold_no_heads:
            ax.stem(x, y, label='Probability', markerfmt='.', basefmt='')
        else:
            ax.stem(x, y, label='Probability', markerfmt='', basefmt='')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax.twinx()
    ax2.set_ylim(-.05, 1.05)
    ax2.plot(x, cumulative, 'tab:red', label='Cumulative')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.legend()
    return fig

def handle(text: str) -> 'tuple[die|None, matplotlib.figure.Figure|None]': # type: ignore
    '''
    text: A math expression involving dice, such as "3d4+7"
    Returns (d, f) where d is a die class object representing the distribution
    of the input expression, and f is a matplotlib figure instance, ie the
    result of _, f = plt.subplots(). If the text expression doesn't return a
    die class object, this returns (None, None).
    Ex:
    handle('3d4')
    plt.savefig('file.png')
    plt.show()

    d, f = handle('4d6')
    f.savefig('4d6.png')
    f.show()
    '''
    x = process_input(text)
    if isinstance(x, die):
        return x, plot(x, text, print_stuff=False)
    else:
        return None, None

def main():
    '''
    Starts an interactive session where the user can type in expressions
    such as 3d4, and the result will be plotted. Only intended for use through
    an interactive Python terminal, using elsewhere may lead to strange results.
    '''
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        try:
            x = process_input(text)
            if isinstance(x, die):
                temp = plot(x, text)
                if temp:
                    print('Plotting in other window. That window must be closed to continue.')
                    temp.show()
            elif isinstance(x, Real):
                print('Numeric result:', x)
                x = None
            if x is None:
                print('Nothing to plot.')
        except NameError:
            print('Not a valid input.')
            traceback.print_exc()
        except Exception:
            print('Error encountered, aborting input.')
            traceback.print_exc()
        exit()
    print('Getting started: Try typing 2d6+3 or 8d6/2 < 12.')
    while True:
        print('\nEnter q to quit. Enter help for options.')
        text = input('>>').lower().strip()
        text = re.sub(r'\s+', ' ', text)
        if text in ('q', 'quit', 'exit'):
            break
        if text in ('?', 'h', 'help'):
            print('Getting started: Try typing 2d6+3 or 8d6/2.')
            print(dice_strings.help_string)
            continue
        if text in ('advanced', 'help advanced', 'h advanced', '?advanced',
                    '? advanced', 'man advanced', 'info advanced', '"help advanced"'):
            print(dice_strings.help_advanced)
            continue
        if text in ('choice', 'help choice', 'h choice', '?choice', '? choice',
                    'man choice', 'info choice', '"help choice"'):
            print(dice_strings.choice_help)
            continue
        if text in ('attack', 'help attack', 'h attack', '?attack', '? attack',
                    'man attack', 'info attack', '"help attack"'):
            print(dice_strings.attack_help)
            continue
        if len(text) > 0 and not text.isspace():
            try:
                x = process_input(text)
                if isinstance(x, die):
                    temp = plot(x, text)
                    if temp:
                        print('Plotting in other window. That window must be closed to continue.')
                        temp.show()
                elif x is None:
                    print('Nothing to plot.')
                else:
                    print('Nothing to plot.\nNumeric result:', x)
            except NameError:
                print('Not a valid input.')
                traceback.print_exc()
            except Exception:
                print('Error encountered, aborting input.')
                traceback.print_exc()
    

if __name__ == '__main__':
    print('\33]0;Dice Script\a', end='')
    sys.stdout.flush()
    main()
