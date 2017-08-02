import numpy as np
import matplotlib.pyplot as plt
from operator import add

small_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
small_times_one = [1.048, 5.275, 9.774, 13.236, 26.54, 29.725, 36.121, 26.375, 55.384, 63.405]
small_times_two = [1.377, 3.137, 8.315, 11.282, 16.63, 14.611, 16.916, 20.301, 30.801, 21.200]
small_times_total = map(add, small_times_one, small_times_two)
small_rules_one = [42, 130, 140, 149, 208, 208, 203, 149, 263, 0]
small_rules_two = [56, 117, 127, 133, 155, 155, 150, 133, 179, 179]

large_data = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
large_times_one = [63.405, 121.365, 202.524, 264.209, 307.326, 365.856, 507.127, 597.276, 659.762, 657.301]
large_times_two = [21.2, 52.692, 96.476, 150.764, 161.281, 226.55, 311.013, 337.46, 353.022, 402.337]
large_times_total = map(add, large_times_one, large_times_two)
large_rules_one = [0, 268, 267, 263, 263, 263, 263, 273, 268, 0]
large_rules_two = [179, 184, 174, 179, 179, 179, 179, 189, 184, 179]

datasets = [10, 20, 50, 100, 250, 500, 1000]
times_one = [1.0480, 5.2750, 32.7360, 63.4050, 160.4340, 340.5430, 657.3010]
times_two = [1.3770, 3.8790, 17.3940, 21.2100, 47.2270, 187.9790, 402.3370]
times_total = map(add, times_one, times_two)
rule_evals = [56, 117, 174, 179, 179, 179, 179]

plt.figure(1)
x = plt.plot(small_data, small_times_one, label='Stage One', linewidth=2.0)
y = plt.plot(small_data, small_times_two, label='Stage Two', linewidth=2.0)
plt.xlabel('Dataset Size')
plt.ylabel('Time (s)')
plt.title('Learning Times for Small Datasets')
plt.legend(loc='upper left')
#plt.show()

plt.figure(2)
x = plt.plot(large_data, large_times_one, label='Stage One', linewidth=2.0)
y = plt.plot(large_data, large_times_two, label='Stage Two', linewidth=2.0)
plt.xlabel('Dataset Size')
plt.ylabel('Time (s)')
plt.title('Learning Times for Large Datasets')
plt.legend(loc='upper left')
plt.show()

plt.figure(3)
x = plt.plot(datasets, times_one, label='Stage One', linewidth=2.0)
y = plt.plot(datasets, times_two, label='Stage Two', linewidth=2.0)
plt.xlabel('Dataset Size')
plt.ylabel('Time (s)')
plt.title('Speed of Learning')
plt.legend(loc='upper left')
#plt.show()
