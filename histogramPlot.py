import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import queue as Queue

# N = 1000
# x = np.random.rand(N)

# plt.hist(x, bins=100)
# plt.xlabel('x')
# plt.ylabel('count')
# plt.show()
# plt.plot(x, '.')
# plt.show()

"""
keys = ['sl', 'sw', 'pl', 'pw', 'c']

k = 0
N = len(keys)
df_iris = pd.read_csv(u'iris.txt', sep=' ')
plt.figure(figsize=(15, 15))

for i, j in product(keys, keys):
    k += 1
    plt.subplot(N, N, k)
    if i == j:
        plt.hist(df_iris[j], bins=20)
        plt.xlabel(j)
    else:
        plt.scatter(df_iris[i],df_iris[j], c=df_iris['c'], cmap='prism')
        plt.xlabel(i)
        plt.ylabel(j)

plt.show()

"""


def find_distance_between_points(p1, p2):
    # Euclidian
    xd = p1[0] - p2[0]
    yd = p1[1] - p2[1]
    x_pow = xd ** 2
    y_pow = yd ** 2
    return (x_pow + y_pow) ** 0.5


def find_closest_k_points(point, points, k):
    dict = {}
    pq = Queue.PriorityQueue()

    for p in range(len(points)):
        distance = find_distance_between_points(x[p], point)
        dict[distance] = x[p]
        pq.put(distance)

    retval = []
    i = 0
    while (i < k):
        retval.append(dict.get(pq.get()))
        i += 1

    return retval


df_iris = pd.read_csv(u'iris.txt', sep=' ')

x = df_iris[['pw', 'pl', 'c']].as_matrix()

# print(x[0])
# print(x[5])
# print(find_distance_between_points(x[0], x[5]))
dict = {}
dict[0.2141] = x[0]
dict[0.2142] = x[1]
dict[0.2142] = x[2]

# print(dict)

q = Queue.PriorityQueue()
q.put(0.2142)
q.put(0.2141)
q.put(0.2143)
q.put(0.2144)

# print(q.get())
# print(q.get())
# print(q.get())
# print(q.get())


print(x[0])
print(find_closest_k_points(x[0], x, 20))
