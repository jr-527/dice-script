This project calculates and plots various probability distributions related to dice. It can handle all of the dice rolls used in 5th edition DnD, along with just about any sensible expression written in [dice notation](https://en.wikipedia.org/wiki/Dice_notation) (leading 1s must be included).

### Features:  
* Easy: You just type something like `6d6+2d4`, press enter, and a graph pops up on screen.
* Fast: Unlike many other projects, my script has no difficulty with relatively large inputs such as 1200d1200. Calculations are done using FFTs and the convolution theorem whenever possible, and the numeric work is generally done using C or vectorized Numpy operations.
* Accurate: My script doesn't use any Monte Carlo simulations or normal approximations. Calculations are generally accurate to 14+ decimal places, although numbers below about 2^-53 run into floating point rounding problems.
* Versatile: This script can perform a wide range of calculations. If Alice rolls 1+1d4 6-sided die, Bob rolls 2d12\*2d4\*0.3 (rounding down), and Carol rolls 3+1d4 and squares the result, then the probability that Alice's total is less than Bob's and Bob's is less than or equal to Carol's can be calculated as follows:  
```(1+1d4)d6 < 2d12*2d4*.3 <= (3+1d4)**2```.

## Installation

Download the repository to your computer. Make sure that you have a fairly modern version of Python installed along with matplotlib and numpy.  
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
## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)
