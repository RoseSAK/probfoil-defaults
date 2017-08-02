import numpy as np
import matplotlib.pyplot as plt
from operator import add

small_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
datasets = [10, 20, 50, 100, 250, 500, 1000]
times_one = [1.0480, 5.2750, 32.7360, 63.4050, 160.4340, 340.5430, 657.3010]
times_two = [1.3770, 3.8790, 17.3940, 21.2100, 47.2270, 187.9790, 402.3370]
times_total = map(add, times_one, times_two)
rule_evals = [56, 117, 174, 179, 179, 179, 179]

plt.figure(1)
x = plt.plot(datasets, times_one, label='Stage One', linewidth=2.0)
y = plt.plot(datasets, times_two, label='Stage Two', linewidth=2.0)
plt.xlabel('Dataset Size')
plt.ylabel('Time (s)')
plt.title('Speed of Learning')
plt.legend(loc='upper left')
plt.show()

plt.figure(2)
