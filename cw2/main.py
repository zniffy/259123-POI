import numpy as np
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# START of KMeans
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
plt.show(block=False)
# END of KMeans

# START of RANSAC
red_plane = []
blue_plane = []
green_plane = []

for i, val in enumerate (pred):
    if val == 0:
        red_plane.append(A[i])
    elif val == 1:
        blue_plane.append(A[i])
    else:
        green_plane.append(A[i])

red_plane_arr = np.array(red_plane)
blue_plane_arr = np.array(blue_plane)
green_plane_arr = np.array(green_plane)


def ransac_method(num_of_inliers, treshold, max_attempts, arr, color=""):
    attempts = 0
    best_inliers_count = 0
    best_attempt = None
    best_normal = None
    num_of_inliers_found = False

    while attempts <= max_attempts:
        idx_arr = np.random.choice(arr.shape[0], size=3, replace=False)
        random_3p = arr[idx_arr]
        A_p = random_3p[0]
        B_p = random_3p[1]
        C_p = random_3p[2]

        vec_A = A_p - C_p
        vec_B = B_p - C_p

        normal_vector = np.cross(vec_A, vec_B)
        uvec_C = normal_vector / np.linalg.norm(normal_vector)

        D = -np.dot(uvec_C, C_p)
        distances = np.abs(np.dot(uvec_C, arr.T) + D)
        inliers_arr = np.where(distances <= treshold)[0]
        inliers_count = len(inliers_arr)

        attempts += 1

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_normal = uvec_C
            best_attempt = attempts

        if best_inliers_count >= num_of_inliers:
            num_of_inliers_found = True

        
        
    if num_of_inliers_found:
        print("Found plane with:", best_inliers_count, "inliers, within:", best_attempt, "attempts")
        print("Best normal vector:", best_normal)
        if color == "r":
            if (best_normal[2] == 1) or (best_normal[2] == -1):
                print("Red plane is horizontal")
            elif (best_normal[0] == 1) or (best_normal[0] == -1) or (best_normal[1] == 1) or (best_normal[1] == -1):
                print("Red plane is vertical")
            else:
                print("Red object is not horizontal nor vertical")
        elif color == "b":
            if (best_normal[2] == 1) or (best_normal[2] == -1):
                print("Blue plane is horizontal")
            elif (best_normal[0] == 1) or (best_normal[0] == -1) or (best_normal[1] == 1) or (best_normal[1] == -1):
                print("Blue plane is vertical")
            else:
                print("Blue object is not horizontal nor vertical")
        elif color == "g":
            if (best_normal[2] == 1) or (best_normal[2] == -1):
                print("Green plane is horizontal")
            elif (best_normal[0] == 1) or (best_normal[0] == -1) or (best_normal[1] == 1) or (best_normal[1] == -1):
                print("Green plane is vertical")
            else:
                print("Green object is not horizontal nor vertical")
        print("######################################################")
    else:
        print("No sufficiently good plane found within", max_attempts, "attempts.")
        print("Maximum inliers found:", best_inliers_count)
        print("######################################################")


ransac_method(20, 1, 100, red_plane_arr, "r")
ransac_method(20, 1, 100, blue_plane_arr, "b")
ransac_method(20, 1, 100, green_plane_arr, "g")
input() # keeps the plot on and lets the program to execute (press Enter in terminal to terminate)