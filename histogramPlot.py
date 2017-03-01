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
    magicNumber = 0.0000000000001
    for p in range(len(points)):
        distance = find_distance_between_points(x[p], point)
        while (distance in dict):
            distance = distance + magicNumber
        dict[distance] = x[p]
        pq.put(distance)

    retval = []
    i = 0
    while (i < k):
        closest_distance = pq.get()
        closest_cell = dict.get(closest_distance)
        retval.append(closest_cell)
        i += 1

    return retval


def predict_element_class(point, points, k):
    closest_points = find_closest_k_points(point, points, k)
    dict = {}
    for i in range(len(closest_points)):
        if (closest_points[i][2] in dict):
            val = dict.get(closest_points[i][2])
            dict[closest_points[i][2]] = val + 1
        else:
            dict[closest_points[i][2]] = 1

    pq = Queue.PriorityQueue()
    if (1.0 in dict):
        pq.put(dict.get(1.0))
    if (2.0 in dict):
        pq.put(dict.get(2.0))
    if (3.0 in dict):
        pq.put(dict.get(3.0))

    while not pq.empty():
        predictedClass = pq.get()

    if (predictedClass == dict.get(1.0)):
        return 1.0
    elif (predictedClass == dict.get(2.0)):
        return 2.0
    else:
        return 3.0


def find_prediction_accuracy(points, k):
    fails = 0
    for p in range(len(points)):
        tmp = np.delete(points, p, 0)
        predicted_class = predict_element_class(points[p], tmp, k)
        expected_class = points[p][2]
        if (predicted_class != expected_class):
            fails = fails + 1
    return ((len(points) - fails) * 100) / len(points)


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

# x = [[1, 1, 1.], [1, 2, 1.], [1, 3, 1.], [1, 4, 2.], [1, 5, 2.], [1, 6, 2.], [1, 7, 3.], [1, 8, 3.], [1, 9, 3.],
#   [1, 10, 3.]]
# [1, 11, 1], [1, 12, 1], [1, 13, 1], [1, 14, 1], [1, 15, 1], [1, 16, 1], [1, 17, 1], [1, 18, 1], [1, 19, 1],
# [1, 20, 1], [1, 21, 1]]

# print(predict_element_class(x[0], x, 1))
# print(predict_element_class(x[1], x, 1))
# print(predict_element_class(x[2], x, 1))
# print(predict_element_class(x[3], x, 1))
# print(predict_element_class(x[4], x, 1))
# print(predict_element_class(x[5], x, 1))
# print(predict_element_class(x[6], x, 1))
# print(predict_element_class(x[7], x, 1))
# print(predict_element_class(x[8], x, 1))
# print(predict_element_class(x[9], x, 1))
# print(find_prediction_accuracy(x,1))
# print(find_prediction_accuracy(x,2))
# print(find_prediction_accuracy(x,3))
# print(find_prediction_accuracy(x,4))

print(find_prediction_accuracy(x, 1))

start = 1
end = 20
bestAcc = {}
bestAcc[1] = 0
accuracyMap = {}
while (start < end):
    accuracy = find_prediction_accuracy(x, start)
    accuracyMap[start] = accuracy
    start = start + 1



print(accuracyMap)
