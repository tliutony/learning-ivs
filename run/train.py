import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from shallowmind.api.train import train

if __name__ == '__main__':
    train()