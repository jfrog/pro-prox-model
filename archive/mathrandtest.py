import random
from math import floor


def test_get_element(arr: list):
    size = len(arr)
    bin_size = 1 / size
    rand_num = random.uniform(0, 1)
    return arr[floor(rand_num / bin_size)]


lista = ["a", "b", "c", "d", "e"]
dict = {"a": 0,
        "b": 0,
        "c": 0,
        "d": 0,
        "e": 0}

# Monte Carlo
for i in range(0, 100000):
    res = test_get_element(lista)
    dict[res] = dict[res] + 1

print(dict)
