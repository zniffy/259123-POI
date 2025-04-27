import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from pyransac3d import Plane


def ransac_method(num_of_inliers, treshold, max_attempts, arr, color=""):
    attempts = 0
    best_inliers_count = 0
    best_attempt = None
    best_normal = None
    best_inliers_acc_arr = None
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
        inliers_acc_arr = arr[inliers_arr]

        attempts += 1

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_normal = uvec_C
            best_attempt = attempts
            best_inliers_acc_arr = inliers_acc_arr
            

        if best_inliers_count >= num_of_inliers:
            num_of_inliers_found = True

        
        
    if num_of_inliers_found:
        print("Found plane with:", best_inliers_count, "inliers, within:", best_attempt, "attempts")
        print("Best normal vector:", best_normal)
        if color == "r":
            if np.all(best_normal == np.array([0, 0, 1])) or np.all(best_normal == np.array([0, 0, -1])):
                print("Red plane is horizontal")
            elif np.all(best_normal == np.array([1, 0, 0])) or np.all(best_normal == np.array([-1, 0, 0])) or np.all(best_normal == np.array([0, 1, 0])) or np.all(best_normal == np.array([0, -1, 0])):
                print("Red plane is vertical")
            else:
                print("Red object is not horizontal nor vertical")
        elif color == "b":
            if np.all(best_normal == np.array([0, 0, 1])) or np.all(best_normal == np.array([0, 0, -1])):
                print("Blue plane is horizontal")
            elif np.all(best_normal == np.array([1, 0, 0])) or np.all(best_normal == np.array([-1, 0, 0])) or np.all(best_normal == np.array([0, 1, 0])) or np.all(best_normal == np.array([0, -1, 0])):
                print("Blue plane is vertical")
            else:
                print("Blue object is not horizontal nor vertical")
        elif color == "g":
            if np.all(best_normal == np.array([0, 0, 1])) or np.all(best_normal == np.array([0, 0, -1])):
                print("Green plane is horizontal")
            elif np.all(best_normal == np.array([1, 0, 0])) or np.all(best_normal == np.array([-1, 0, 0])) or np.all(best_normal == np.array([0, 1, 0])) or np.all(best_normal == np.array([0, -1, 0])):
                print("Green plane is vertical")
            else:
                print("Green object is not horizontal nor vertical")

        centroid = np.mean(best_inliers_acc_arr, axis=0)
        centered_points = best_inliers_acc_arr - centroid
        U, S, V = np.linalg.svd(centered_points)
        final_normal = V[-1]
        final_normal_u = final_normal / np.linalg.norm(final_normal)
        print("Final normal vector found using least squares method is:", final_normal_u)
        print("###################################################################################################")
    else:
        print("No sufficiently good plane found within", max_attempts, "attempts.")
        print("Maximum inliers found:", best_inliers_count)
        print("###################################################################################################")

# START of KMeans
input_files = ["cw1/cylinder.xyz", "cw1/horizontal.xyz", "cw1/vertical.xyz"]

with open('cw2/merged.xyz', 'w', newline='\n') as outfile:
    for filename in input_files:
        with open(filename) as infile:
            content = infile.read()
            outfile.write(content)

#def dbscan_method(X):
    #db = DBSCAN(eps=2.5, min_samples=50).fit(X)
    #labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #n_noise_ = list(labels).count(-1)

    #print("Estimated number of clusters: %d" % n_clusters_)
    #print("Estimated number of noise points: %d" % n_noise_)


def process_point_cloud_with_dbscan_ransac(arr, eps=0, min_samples=2, ransac_thresh=0.1, max_iterations=100):
    """
    Processes a point cloud by first clustering with DBSCAN and then
    fitting planes to each cluster using pyransac3d.

    Args:
        arr (np.ndarray): A (N, 3) array of 3D point coordinates.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other (DBSCAN parameter).
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point (DBSCAN parameter).
        ransac_thresh (float): The maximum distance for a point to be considered an inlier to a plane (pyransac3d parameter).
        max_iterations (int): The maximum number of iterations to run RANSAC (pyransac3d parameter).
    """
    # 1. Cluster the point cloud using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(arr)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"Estimated number of clusters: {n_clusters}")

    # 2. Process each cluster (excluding noise points labeled as -1)
    for cluster_id in range(n_clusters):
        cluster_points = arr[clusters == cluster_id]
        print(f"\nProcessing cluster {cluster_id} with {len(cluster_points)} points.")

        if len(cluster_points) >= 3:
            # 3. Fit a plane to the cluster using pyransac3d
            plane_fitter = Plane()
            best_plane, inliers = plane_fitter.fit(cluster_points, thresh=ransac_thresh, max_iterations=max_iterations)

            if best_plane is not None:
                normal_vector = best_plane[0:3] / np.linalg.norm(best_plane[0:3])
                print("Fitted plane parameters (a, b, c, d):", best_plane)
                print("Normal vector:", normal_vector)

                # 4. Determine if the plane is approximately vertical or horizontal
                tolerance = 0.01
                abs_normal = np.abs(normal_vector)
                is_close_to_zero = abs_normal < tolerance
                is_close_to_one = np.abs(abs_normal - 1.0) < tolerance

                is_vertical = is_close_to_one[2] and is_close_to_zero[0] and is_close_to_zero[1]
                is_horizontal_x = is_close_to_one[0] and is_close_to_zero[1] and is_close_to_zero[2]
                is_horizontal_y = is_close_to_one[1] and is_close_to_zero[0] and is_close_to_zero[2]
                is_horizontal = is_horizontal_x or is_horizontal_y

                if is_horizontal:
                    print("Cluster is approximately horizontal.")
                elif is_vertical:
                    print("Cluster is approximately vertical.")
                else:
                    print("Cluster is neither approximately horizontal nor vertical.")
            else:
                print("Could not fit a plane to this cluster using RANSAC.")
        else:
            print("Cluster has fewer than 3 points, skipping plane fitting.")

    # Handle noise points if any
    noise_points = arr[clusters == -1]
    if len(noise_points) > 0:
        print(f"\nFound {len(noise_points)} noise points.")



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


ransac_method(20, 1, 100, red_plane_arr, "r")
ransac_method(20, 1, 100, blue_plane_arr, "b")
ransac_method(20, 1, 100, green_plane_arr, "g")

#dbscan_method(A)
process_point_cloud_with_dbscan_ransac(A, eps=0.1, min_samples=50, ransac_thresh=0.2, max_iterations=200)
input() # keeps the plot on and lets the program to execute (press Enter in terminal to terminate)