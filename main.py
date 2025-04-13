import numpy as np

from numpy import random
from aug_ablation import *

train('acm', t=2, tao=0.5, k=50, rc=-0.3, kk=50, lr=1e-3,la=2.0)
