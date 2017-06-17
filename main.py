from sys import argv
from backprop import train
from backprop import test

if __name__ == "__main__":
    if len(argv) == 1:
        pass
    elif argv[1] == 'train':
        train()
    elif argv[1] == 'test':
        test()
    else:
        pass
