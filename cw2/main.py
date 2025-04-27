import numpy as np
import csv
import matplotlib.pyplot as plt
import pyransac3d as pyrsc

from sklearn.cluster import KMeans, DBSCAN


def ransac_method(num_of_inliers, treshold, max_attempts, arr, color=""):
    attempts = 0                                                        # attempts counter
    best_inliers_count = 0                                              # counter of maximum found number of inliers
    best_attempt = None                                                 # indicator at what attempt the best_inliers_count occured
    best_normal = None                                                  # normal vector found for the maximum found number of inliers
    best_inliers_acc_arr = None                                         # array of inliers found for the maximum number of inliers
    num_of_inliers_found = False                                        # flag to check if expected num_of_inliers were found

    while attempts <= max_attempts:
        idx_arr = np.random.choice(arr.shape[0], size=3, replace=False) # randomly chosen indexes from cloud of points
        random_3p = arr[idx_arr]                                        # actual chosen points from cloud of points
        A_p = random_3p[0]                                             
        B_p = random_3p[1]
        C_p = random_3p[2]

        vec_A = A_p - C_p                                               # calculating vector of point A
        vec_B = B_p - C_p                                               # calculating vector of point B

        normal_vector = np.cross(vec_A, vec_B)                          # calculating vector of point C
        uvec_C = normal_vector / np.linalg.norm(normal_vector)          # unit vector C

        D = -np.dot(uvec_C, C_p)                                        # calculating distance D
        distances = np.abs(np.dot(uvec_C, arr.T) + D)                   # calculating distances of all points to the vec C
        inliers_arr = np.where(distances <= treshold)[0]                # indexes of points from input array that matched treshold 
        inliers_count = len(inliers_arr)                                # number of found inliers
        inliers_acc_arr = arr[inliers_arr]                              # actual cloud points that are inliers

        attempts += 1                                                   # incrementing attempts

        if inliers_count > best_inliers_count:                          # if statement for remembering the best values
            best_inliers_count = inliers_count
            best_normal = uvec_C
            best_attempt = attempts
            best_inliers_acc_arr = inliers_acc_arr
            

        if best_inliers_count >= num_of_inliers:                        
            num_of_inliers_found = True                                 # changing value of the flag to inform that min. number of inliers were found

        
    if num_of_inliers_found:
        print("Found plane with:", best_inliers_count, "inliers, at:", best_attempt, "attempt, for manual RANSCAN method + KMeans")
        print("Best normal vector:", best_normal)
        if color == "r":
            if np.all(best_normal == np.array([0, 0, 1])) or np.all(best_normal == np.array([0, 0, -1])):
                print("Red plane is horizontal")
            elif np.all(best_normal == np.array([1, 0, 0])) or np.all(best_normal == np.array([-1, 0, 0])) or \
                np.all(best_normal == np.array([0, 1, 0])) or np.all(best_normal == np.array([0, -1, 0])):
                print("Red plane is vertical")
            else:
                print("Red object is not horizontal nor vertical")
        elif color == "b":
            if np.all(best_normal == np.array([0, 0, 1])) or np.all(best_normal == np.array([0, 0, -1])):
                print("Blue plane is horizontal")
            elif np.all(best_normal == np.array([1, 0, 0])) or np.all(best_normal == np.array([-1, 0, 0])) or \
                np.all(best_normal == np.array([0, 1, 0])) or np.all(best_normal == np.array([0, -1, 0])):
                print("Blue plane is vertical")
            else:
                print("Blue object is not horizontal nor vertical")
        elif color == "g":
            if np.all(best_normal == np.array([0, 0, 1])) or np.all(best_normal == np.array([0, 0, -1])):
                print("Green plane is horizontal")
            elif np.all(best_normal == np.array([1, 0, 0])) or np.all(best_normal == np.array([-1, 0, 0])) or \
                np.all(best_normal == np.array([0, 1, 0])) or np.all(best_normal == np.array([0, -1, 0])):
                print("Green plane is vertical")
            else:
                print("Green object is not horizontal nor vertical")

        # Calculating final normal vector using least square method
        centroid = np.mean(best_inliers_acc_arr, axis=0)
        centered_points = best_inliers_acc_arr - centroid
        U, S, V = np.linalg.svd(centered_points)
        final_normal = V[-1]
        final_normal_u = final_normal / np.linalg.norm(final_normal)

        print("Final normal vector found using least squares method is:", final_normal_u)
        print("##################################################################################################################\n")
    else:
        if color == "r":
            print("No sufficiently good plane found within", max_attempts, "attempts for red object, for manual RANSAC method + KMeans.")
        elif color == "b":
            print("No sufficiently good plane found within", max_attempts, "attempts for blue object, for manual RANSAC method + KMeans.")
        elif color == "g":
            print("No sufficiently good plane found within", max_attempts, "attempts for green object, for manual RANSAC method + KMeans.")
        print("Maximum inliers found:", best_inliers_count)
        print("##################################################################################################################\n")


def ransac_method_pyransac3d(num_of_inliers, treshold, max_attempts, arr, color=""):
    normal_vector, inliers_arr = pyrsc.Plane().fit(arr, treshold, num_of_inliers, max_attempts)
    normal_vector_u = normal_vector[0:3] / np.linalg.norm(normal_vector[0:3])
    inliers_acc_arr = arr[inliers_arr]                                                              # actual cloud points that are inliers
        
    if len(inliers_arr) >= num_of_inliers:
        print("Found plane with:", len(inliers_arr), "inliers, for build-in RANSAC method + DBSCAN")
        print("Best normal vector:", normal_vector_u)
        if color == "r":
            if np.all(normal_vector_u == np.array([0, 0, 1])) or np.all(normal_vector_u == np.array([0, 0, -1])):
                print("Red plane is horizontal")
            elif np.all(normal_vector_u == np.array([1, 0, 0])) or np.all(normal_vector_u == np.array([-1, 0, 0])) or \
                np.all(normal_vector_u == np.array([0, 1, 0])) or np.all(normal_vector_u == np.array([0, -1, 0])):
                print("Red plane is vertical")
            else:
                print("Red object is not horizontal nor vertical")
        elif color == "b":
            if np.all(normal_vector_u == np.array([0, 0, 1])) or np.all(normal_vector_u == np.array([0, 0, -1])):
                print("Blue plane is horizontal")
            elif np.all(normal_vector_u == np.array([1, 0, 0])) or np.all(normal_vector_u == np.array([-1, 0, 0])) or \
                np.all(normal_vector_u == np.array([0, 1, 0])) or np.all(normal_vector_u == np.array([0, -1, 0])):
                print("Blue plane is vertical")
            else:
                print("Blue object is not horizontal nor vertical")
        elif color == "g":
            if np.all(normal_vector_u == np.array([0, 0, 1])) or np.all(normal_vector_u == np.array([0, 0, -1])):
                print("Green plane is horizontal")
            elif np.all(normal_vector_u == np.array([1, 0, 0])) or np.all(normal_vector_u == np.array([-1, 0, 0])) or np.all(normal_vector_u == np.array([0, 1, 0])) or np.all(normal_vector_u == np.array([0, -1, 0])):
                print("Green plane is vertical")
            else:
                print("Green object is not horizontal nor vertical")

        # Calculating final normal vector using least square method
        centroid = np.mean(inliers_acc_arr, axis=0)
        centered_points = inliers_acc_arr - centroid
        U, S, V = np.linalg.svd(centered_points)
        final_normal = V[-1]
        final_normal_u = final_normal / np.linalg.norm(final_normal)

        print("Final normal vector found using least squares method is:", final_normal_u)
        print("******************************************************************************************************************\n")
    else:
        if color == "r":
            print("No sufficiently good plane found within", max_attempts, "attempts for red object, for build-in RANSAC method + DBSCAN.")
        elif color == "b":
            print("No sufficiently good plane found within", max_attempts, "attempts for blue object, for build-in RANSAC method + DBSCAN.")
        elif color == "g":
            print("No sufficiently good plane found within", max_attempts, "attempts for green object, for build-in RANSAC method + DBSCAN.")
        print("*****************************************************************************************************************\n")

    

# MERGE and FILL UP array with values ###################################################
input_files = ["cw1/cylinder.xyz", "cw1/horizontal.xyz", "cw1/vertical.xyz"]            #
                                                                                        #
with open('cw2/merged.xyz', 'w', newline='\n') as outfile:                              #
    for filename in input_files:                                                        #
        with open(filename) as infile:                                                  #
            content = infile.read()                                                     #
            outfile.write(content)                                                      #
                                                                                        #
                                                                                        #
points = []                                                                             #
                                                                                        #
with open('cw2/merged.xyz', 'r') as file:                                               #
    reader = csv.reader(file, delimiter=',')                                            #
    for row in reader:                                                                  #
        points.append([float(row[0]), float(row[1]), float(row[2])])                    #
                                                                                        #
                                                                                        #
A = np.array(points)                                                                    #
#########################################################################################


# START of KMeans #######################################################################
clusters = KMeans(n_clusters=3)                                                         #
pred = clusters.fit_predict(A)                                                          #
                                                                                        #
r_kmeans = pred == 0                                                                    #
b_kmeans = pred == 1                                                                    #
g_kmeans = pred == 2                                                                    #
                                                                                        #
fig1 = plt.figure()                                                                     #
ax = fig1.add_subplot(projection='3d')                                                  #
                                                                                        #
ax.scatter(A[r_kmeans, 0], A[r_kmeans, 1], A[r_kmeans, 2], c='r', marker='.')           #
ax.scatter(A[b_kmeans, 0], A[b_kmeans, 1], A[b_kmeans, 2], c='b', marker='.')           #
ax.scatter(A[g_kmeans, 0], A[g_kmeans, 1], A[g_kmeans, 2], c='g', marker='.')           #
ax.set_xlabel('X')                                                                      #
ax.set_ylabel('Y')                                                                      #
ax.set_zlabel('Z')                                                                      #
ax.set_title('3D KMeans Clustering')                                                    #
plt.show(block=False)                                                                   #
                                                                                        #                                                    
red_plane_kmeans = []                                                                   #
blue_plane_kmeans = []                                                                  #
green_plane_kmeans = []                                                                 #
                                                                                        #
for i, val in enumerate (pred):                                                         #
    if val == 0:                                                                        #
        red_plane_kmeans.append(A[i])                                                   #
    elif val == 1:                                                                      #
        blue_plane_kmeans.append(A[i])                                                  #
    else:                                                                               #
        green_plane_kmeans.append(A[i])                                                 #
                                                                                        #
red_plane_arr_KM = np.array(red_plane_kmeans)                                           #
blue_plane_arr_KM = np.array(blue_plane_kmeans)                                         #
green_plane_arr_KM = np.array(green_plane_kmeans)                                       #
# END of KMeans #########################################################################


# START of DBSCAN #######################################################################
db = DBSCAN(eps=1.0, min_samples=10).fit(A)                                             #
labels = db.labels_                                                                     #
                                                                                        #   
r_dbscan = labels == 0                                                                  #                                                                   
b_dbscan = labels == 1                                                                  #  
g_dbscan = labels == 2                                                                  #
                                                                                        #
fig2 = plt.figure()                                                                     #
axx = fig2.add_subplot(projection='3d')                                                 #
                                                                                        #
axx.scatter(A[r_dbscan, 0], A[r_dbscan, 1], A[r_dbscan, 2], c='r', marker='.')          #                       
axx.scatter(A[b_dbscan, 0], A[b_dbscan, 1], A[b_dbscan, 2], c='b', marker='.')          #            
axx.scatter(A[g_dbscan, 0], A[g_dbscan, 1], A[g_dbscan, 2], c='g', marker='.')          #            
axx.set_xlabel('X')                                                                     #
axx.set_ylabel('Y')                                                                     #
axx.set_zlabel('Z')                                                                     #
axx.set_title('3D DBSCAN Clustering')                                                   #                                                                           
plt.show(block=False)                                                                   #
                                                                                        #
red_plane_dbscan = []                                                                   #
blue_plane_dbscan = []                                                                  #
green_plane_dbscan = []                                                                 #
                                                                                        #
for i, val in enumerate (labels):                                                       #  
    if val == 0:                                                                        #
        red_plane_dbscan.append(A[i])                                                   #                                                  
    elif val == 1:                                                                      #
        blue_plane_dbscan.append(A[i])                                                  #
    else:                                                                               #
        green_plane_dbscan.append(A[i])                                                 #
                                                                                        #
red_plane_arr_DB = np.array(red_plane_dbscan)                                           #
blue_plane_arr_DB = np.array(blue_plane_dbscan)                                         #
green_plane_arr_DB = np.array(green_plane_dbscan)                                       #
# END of DBSCAN #########################################################################


'''
# Only for DEBBUG
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
'''


ransac_method(20, 1, 100, red_plane_arr_KM, "r")
ransac_method(20, 1, 100, blue_plane_arr_KM, "b")
ransac_method(20, 1, 100, green_plane_arr_KM, "g")


ransac_method_pyransac3d(20, 1, 100, red_plane_arr_DB, "r")
ransac_method_pyransac3d(20, 1, 100, blue_plane_arr_DB, "b")
ransac_method_pyransac3d(20, 1, 100, green_plane_arr_DB, "g")


input("Press ENTER in terminal to terminate the code")
