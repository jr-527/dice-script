This project calculates and plots various probability distributions related to dice. It can handle all of the dice rolls used in 5th edition DnD, along with just about any sensible expression written in [dice notation](https://en.wikipedia.org/wiki/Dice_notation) (leading 1s must be included).

### Features:  
* Easy: You just type something like `6d6+2d4`, press enter, and a graph pops up on screen.
* Fast: Unlike many other projects, my script has no difficulty with relatively large inputs such as 1200d1200. Calculations are done using FFTs and the convolution theorem whenever possible, and the numeric work is generally done using C or vectorized Numpy operations.
* Accurate: My script doesn't use any Monte Carlo simulations or normal approximations. Calculations are generally accurate to 14+ decimal places, although numbers below about 2^-53 run into floating point rounding problems.
* Versatile: This script can perform a wide range of calculations. If Alice rolls 1+1d4 6-sided die, Bob rolls 2d12\*2d4\*0.3 (rounding down), and Carol rolls 3+1d4 and squares the result, then the probability that Alice's total is less than Bob's and Bob's is less than or equal to Carol's can be calculated as follows:  
```(1+1d4)d6 < 2d12*2d4*.3 <= (3+1d4)**2```.
* Easy "API": Comes with an incredibly simple interface which can be easily integrated with other Python code. 

## Installation

Download the repository to your computer. Make sure that you have Python 3.10 or above installed along with matplotlib and numpy.  
This repository uses code written in C to speed up a few operations. I have provided compiled 64 bit .dll and .so files that work on my version of Windows 11 and Ubuntu through WSL2. If they don't work properly on your machine, my Python code falls back to a marginally slower Python implementation and prints a line saying as much on startup.

## Usage

To get started, run dice.py through a command line. If you are on Windows, you can instead double-click on windows.bat. You might have to write python3 instead of python.
```
python ./dice.py
```
Further instructions will then be displayed.

You can also run things through the command line arguments
```
python ./dice.py 3d4 - 2
python ./dice.py 3d4-2
```
Another option is to `import dice` in another Python file, which is explained in the next section.

## API

If dice.py is run through the command line, it acts as its own REPL environment of sorts. If you instead want to integrate it into other Python code, do this:
```
import dice
d, fig = dice.handle('3d6*2d4')
```
`fig` is a Matplotlib figure showing a plot of the [pmf](https://en.wikipedia.org/wiki/Probability_mass_function) and
[cdf](https://en.wikipedia.org/wiki/Cumulative_distribution_function) of the distribution, so you can do things like `fig.savefig('filename.png')` or `fig.show()` with it.  
`d` is a die object. `d.arr` is a numpy array such that `d.arr[x-d.start]` is the probability of the specified dice roll being equal to `x`.  
`d` can also be sliced, so `d[5]` is the probability of the dice roll equalling 5 and `d[5:8]` gives `np.array([d[5], d[6], d[7]])`.  
You can perform arithmetic using die objects; `2 * dice.handle('3d6')[0] * dice.handle('2d4')[0] + 3` is equivalent to `dice.handle('2 * 3d6 * 2d4 + 3')[0]`. I recommend
placing die arithmetic inside the handle function if possible, because that allows you to use more powerful features related to inequalities and whatnot, should have less overhead, and makes plotting simpler.  
By default, the handle function doesn't print anything to the console. If you want it to print some statistics to the console, use `dice.handle(text, False)`

## License

[MIT License](https://opensource.org/license/mit)
