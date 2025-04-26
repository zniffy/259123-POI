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
plt.show()
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

idx_random_3p_red = np.random.choice(red_plane_arr.shape[0], size=3, replace=False)
idx_random_3p_blue = np.random.choice(blue_plane_arr.shape[0], size=3, replace=False)
idx_random_3p_green = np.random.choice(green_plane_arr.shape[0], size=3, replace=False)

random_3p_red = red_plane_arr[idx_random_3p_red]
random_3p_blue = blue_plane_arr[idx_random_3p_blue]
random_3p_green = green_plane_arr[idx_random_3p_green]

#print(red_plane_arr)
#print(random_3p_red)

def ransac_method(num_of_inliers, treshold, max_attempts, arr):
    attempts = 0
    inliers = 0

    while attempts <=  max_attempts:
        idx_arr = np.random.choice(arr.shape[0], size=3, replace=False)
        random_3p = arr[idx_arr]
        A_p = random_3p[0]
        B_p = random_3p[1]
        C_p = random_3p[2]

        vec_A = A_p - C_p
        vec_B = B_p - C_p

        uvec_A = vec_A/np.linalg.norm(vec_A)
        uvec_B = vec_B/np.linalg.norm(vec_B)
        uvec_C = np.cross(uvec_A, uvec_B)

        
        D = -np.sum(np.multiply(uvec_C, C_p))
        distance_all_p = (uvec_C*arr + D)/np.linalg.norm(uvec_C)
        inliers_arr = np.where(np.abs(distance_all_p <= treshold))[0]
        inliers = len(inliers_arr)

        attempts += 1

        if inliers >= num_of_inliers:
            print("found answer with:", inliers, "inliers, within:", attempts, "attempts")
            break;

    if attempts > max_attempts:
        print("No answer")
        print("Managed to find", inliers, "inliers")
        


ransac_method(300, 0.0001, 100, red_plane_arr)