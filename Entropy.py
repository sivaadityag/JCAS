import math

p = 0.25

entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

capacity = 1 - entropy
