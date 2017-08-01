import numpy as np
import matplotlib.pyplot as plt
from operator import add

datasets = [10, 20, 50, 100, 250, 500]
times_one = [1.0480, 5.2750, 32.7360, 63.4050, 160.4340, 340.5430]
times_two = [1.3770, 3.8790, 17.3940, 21.2100, 47.2270, 187.9790]
times_total = map(add, times_one, times_two)
print(times_total)
rule_evals = [56, 117, 174, 179, 179, 179]

plt.plot(datasets, times_one)
plt.plot(datasets, times_two)
plt.xlabel('Dataset Size')
plt.ylabel('Time (s)')
plt.title('Speed of Learning')
plt.legend()
plt.show()
