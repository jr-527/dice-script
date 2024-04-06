from die import die, ndm, is_number
import numpy as np
from dice_functions import min0, min1, min_val, mean, var, sd, order_stat, order, highest, adv, advantage, lowest, disadv, dis, disadvantage, choice, attack, crit, check, save, reroll, sample, multiple_inequality, drop
from round_to_width import round_to_width as round_w
import dice_strings
import os
import sys
import numpy as np
plt = None # For asynchronous importing, to start up more quickly
import threading
import re
import warnings
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
print('\33]0;Dice Script\a', end='')
sys.stdout.flush()

def process_input(text):
    '''
    Internal function. Processes then calls eval on text, plotting if possible.
    '''
    new_text = re.sub('\s', ' ', text)
    new_text = re.sub(r'[)]d', r')@1d', new_text)
    drop_regexp = r'([1-9][0-9]*)d([1-9][0-9]*)\s*(kh|dh|kl|dl|keep lowest|' + \
        r'keep highest|drop lowest|drop highest)\s*([1-9][0-9]*)?'
    new_text = re.sub(drop_regexp, r'drop(\1, \2, "\3", \4)', new_text)
    new_text = re.sub(r'\^', '**', new_text)
    new_text = re.sub(r'([1-9][0-9]*)d([1-9][0-9]*)',
        r'die(ndm(\1,\2),\1,"\1d\2",True)', new_text)
    # Python's parse rules mean eval can't properly process 2 < 1d20 < 19
    nts = re.split(r'(\>=|\>|\<=|\<|==|!=)', new_text) # "new text split"
    relations = ['>', '<', '>=', '<=', '==', '!=']
    while True:
        has_changed = False
        for i in range(len(nts)-1, -1, -1):
            if nts[i] in relations:
                lhs, op, rhs = nts[i-1], nts[i], nts[i+1]
                if not parens_balanced(lhs) and not parens_balanced(rhs):
                    if parens_balanced(lhs+op+rhs):
                        nts[i-1:i+2] = [lhs+op+rhs]
                        i -= 1
                        has_changed = True
        if not has_changed:
            break
    if len(nts) > 3:
        try:
            nts = [eval(x) if x not in relations else x for x in nts]
            p = multiple_inequality(*nts)
            x = die(np.array([1-p, p]), 0, text, True, True)
        except SyntaxError:
            x = eval(new_text)
    else:
        x = eval(new_text)
    return x

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
    if hasattr(d, '__len__'):
        d = die(d, 0)
    if d.isProbability:
        print('Probability:',
            f"{round_w(mean(d),15,'left',leading_zero=True).strip()}",
            'standard deviation:',
            f"{round_w(sd(d),15,'left',leading_zero=True).strip()}")
        s = int(sample(d))
        s = (f'{s} (Succeed)') if s else (f'{s} (Fail)')
        print(f'Random sample from distribution:', s)
        if d.arr[0] == 0 or d.arr[0] == 1:
            print('Nothing to plot.')
            return
        print('Plotting in other window. That window must be closed to continue.')
        fig, ax = plt.subplots()
        if name:
            fig.canvas.manager.set_window_title(name)
            plt.title(name)
        ax.bar(0, d.arr[0], 0.4, bottom=d.arr[1], label='Fail', color='tab:red')
        ax.bar(0, d.arr[1], 0.4, bottom=0, label='Succeed', color='tab:blue')
        ax.set_xlim(-1,1)
        ax.get_xaxis().set_ticks([])
        ax.legend()
        plt.show()
        return
    print('Mean:',
        f"{round_w(mean(d),15,'left',leading_zero=True).strip()}",
        'standard deviation:',
        f"{round_w(sd(d),15,'left',leading_zero=True).strip()}")
    print(f'Random sample from distribution: {sample(d)}')
    if len(d.arr) == 1:
        print('Nothing to plot.')
        return
    print('Plotting in other window. That window must be closed to continue.')
    fig, ax = plt.subplots()
    if name:
        if '=' in name or '<' in name or '>' in name:
            name = d.name
        fig.canvas.manager.set_window_title(name)
        plt.title(name)
    y = d.arr
    cumulative = np.cumsum(y)
    threshold_small_heads = 100
    threshold_no_heads = 1000
    threshold_continuous = 10000
    if len(y) >= threshold_continuous:
        col_size = int(len(y)/(threshold_continuous/2))
        # reshape y to (col_size, ?)
        new_cols = int(np.ceil(len(y)/col_size))
        y = np.max( # resample by taking the max of every 5000 elements
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
    plt.show()

def parens_balanced(string):
    net = 0
    for char in string:
        if char == '(':
            net += 1
        elif char == ')':
            net -= 1
            if net < 0:
                return False
    return net == 0

if __name__ == '__main__':
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        plot(process_input(text), text)
        exit()
    print('Getting started: Try typing 2d6+3 or 8d6/2.')
    while True:
        print('\nEnter q to quit. Enter help for options.')
        text = input('>>').lower()
        if text in ('q', 'quit', 'exit'):
            exit()
        if text in ('?', 'h', 'help'):
            print('Getting started: Try typing 2d6+3 or 8d6/2.')
            print(dice_strings.help_string)
            continue
        if text in ('advanced', 'help advanced', 'h advanced', '?advanced', '? advanced', '"help advanced"'):
            print(dice_strings.help_advanced)
            continue
        if text in ('choice', 'help choice', 'h choice', '?choice', '? choice', '"help choice"'):
            print(dice_strings.choice_help)
            continue
        if text in ('attack', 'help attack', 'h attack', '?attack', '? attack', '"help attack"'):
            print(dice_strings.attack_help)
            continue
        if len(text) > 0 and not text.isspace():
            try:
                x = process_input(text)
                if isinstance(x, (np.ndarray, list)):
                    # This won't happen under normal use but it's helpful for
                    # development reasons.
                    print(x)
                if is_number(x) or isinstance(x, np.bool_):
                    print('Numeric result:', x)
                    x = None
                if x is None:
                    print('Nothing to plot.')
                else:
                    plot(x, text)
            except NameError as e:
                print('Not a valid input.')
                traceback.print_exc()
            except Exception as e:
                print('Error encountered, aborting input.')
                traceback.print_exc()
