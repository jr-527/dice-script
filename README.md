This project finds and plots the probability mass functions of various things that you can do using dice. It's designed around the dice rolls used in 5th edition DnD. This project uses [dice notation](https://en.wikipedia.org/wiki/Dice_notation) (leading 1s must be included).  
I've seen many implementations of this and they're generally some combination of unwieldy, inefficient, inaccurate, and limited, often using large Monte Carlo simulations or enumerating the entire sample space. I tried my best to make this code easy to use, reasonably efficient, accurate, and capable of performing a wide variety of computations, making use of the convolution theorem and FFTs whenever possible.
For example, if you input the text ```15d12 * 3 + 3d4 * (2d6 - 1d10)``` this project can calculate and plot the PMF of that expression, accurate to 14+ decimal places, without any perceptible delay.

## Installation

Download the repository to your computer. Make sure that your Python installation has matplotlib and numpy, and that you're running a fairly modern version of Python.  
This repository uses code written in C to speed up a few operations. I have provided compiled 64 bit .dll and .so files that work on my version of Windows 11 and Ubuntu through WSL2. If they don't work properly on your machine, my Python code falls back to a marginally slower Python implementation and prints a line saying as much on startup.

## Usage

To get started, run dice.py through a command line. You might have to write python3 instead of python.
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
