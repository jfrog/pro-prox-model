import random
import matplotlib.pyplot as plt


def make_unif_dist(x: int, y: int):
    size = y - x
    bin_size = 1 / size
    rand_num = random.uniform(0, 1)
    return (rand_num / bin_size) + x


list_of_vals = []
for i in range(0, 10000):
    list_of_vals.append(make_unif_dist(0, 100))

graph = plt.hist(list_of_vals, bins='auto')
plt.show()
