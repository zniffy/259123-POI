import numpy as np
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_files = ["cw1/cylinder.xyz", "cw1/horizontal.xyz", "cw1/vertical.xyz"]

with open('cw2/merged.xyz', 'w', newline='\n') as outfile:
    for filename in input_files:
        with open(filename) as infile:
            content = infile.read()
            outfile.write(content)


points = []
with open('cw2/merged.xyz', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        points.append([float(row[0]), float(row[1]), float(row[2])])


A = np.array(points)
clusters = KMeans(n_clusters=3)
pred = clusters.fit_predict(A)

r = pred == 0
b = pred == 1
g = pred == 2 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(A[r, 0], A[r, 1], A[r, 2], c='r')
ax.scatter(A[b, 0], A[b, 1], A[b, 2], c='b')
ax.scatter(A[g, 0], A[g, 1], A[g, 2], c='g')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D KMeans Clustering')
ax.legend()
plt.show()
