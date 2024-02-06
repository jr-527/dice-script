from die import die, ndm, is_number
import numpy as np
from dice_functions import min0, min1, min_val, mean, var, sd, order_stat, order, highest, adv, advantage, lowest, disadv, dis, disadvantage, choice, attack, crit, check, save, reroll, sample
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
    new_text = re.sub('(4d6dl)|(4d6 drop lowest)', 'stat_roll()', new_text)
    new_text = re.sub(r'\^', '**', new_text)
    new_text = re.sub(r'([1-9][0-9]*)d([1-9][0-9]*)',
        r'die(ndm(int(\1),int(\2)),int(\1),"\1d\2", True)', new_text)
    x = eval(new_text)
    if is_number(x) or isinstance(x, np.bool_):
        print('Numeric result:', x)
        x = None
    if x is None:
        print('Nothing to plot.')
    else:
        print('Plotting in other window. That window must be closed to continue.')
        plot(x, text)

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

def stat_roll(): # Needs to have a different name than say 4d6dl() bc regex
    arr = np.array(
        [1, 4, 10, 21, 38, 62, 91, 122, 148, 167, 172, 160, 131, 94, 54, 21],
        dtype=np.float64
    )
    arr /= 1296
    return die(arr, 3, '4d6dl', True)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        process_input(' '.join(sys.argv[1:]))
        exit()
    # print(help_string)
    print('Getting started: Try typing 2d6+3 or 8d6/2.')
    while True:
        print('\nEnter q to quit. Enter help for options.')
        text = input('>>')
        if text.lower() in ('q', 'quit', 'exit'):
            exit()
        if text.lower() in ('?', 'h', 'help'):
            print('Getting started: Try typing 2d6+3 or 8d6/2.')
            print(dice_strings.help_string)
            continue
        if text.lower() in ('help advanced', 'h advanced', '?advanced', '? advanced', '"help advanced"'):
            print(dice_strings.help_advanced)
            continue
        if text.lower() in ('help choice', 'h choice', '?choice', '? choice', '"help choice"'):
            print(dice_strings.choice_help)
            continue
        if text.lower() in ('help attack', 'h attack', '?attack', '? attack', '"help attack"'):
            print(dice_strings.attack_help)
            continue
        if len(text) > 0 and not text.isspace():
            try:
                process_input(text)
            except NameError as e:
                print('Not a valid input.')
                traceback.print_exc()
            except Exception as e:
                print('Error encountered, aborting input.')
                traceback.print_exc()
