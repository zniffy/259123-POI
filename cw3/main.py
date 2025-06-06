from PIL import Image
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
import skimage
import os
import random
import csv
import pandas
import matplotlib.pyplot as plt

def slice_image_randomly(image_path, output_folder, slice_size=128, num_slices=10):
    add_zeroes = len(str(abs(num_slices))) #additional lineto help with naming slices
    name_only = os.path.splitext(os.path.basename(image_path))[0] #gets name of actual category from path string

    #deleting all of the previous slices
    for filename_to_del in os.listdir(output_folder):
        file_path_to_del = os.path.join(output_folder, filename_to_del)
        if os.path.isfile(file_path_to_del):
            os.remove(file_path_to_del)

    image = Image.open(image_path) #loading the image
    width, height = image.size #size of image

    #checking if the size of a picture isn't to small in reference to the slice size
    if width < slice_size or height < slice_size:
        raise ValueError("Image is smaller than the slice size.")

    #max "coordinates" of start point of slice where it would still fit
    max_x = width - slice_size
    max_y = height - slice_size

    #set of random positions
    positions = set()
    while len(positions) < num_slices:
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        positions.add((x, y))

    #cropping picture and saving slcies
    for i, (x, y) in enumerate(positions):
        crop_img = (x, y, x + slice_size, y + slice_size)
        slice_img = image.crop(crop_img)
        slice_filename = f"{name_only}_slice_{i+1:0{add_zeroes}d}.jpg"
        slice_path = os.path.join(output_folder, slice_filename)
        slice_img.save(slice_path)

    print(f"{len(positions)} random slices saved in {output_folder}")


def glcm_features(image_path, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
    #loading the image -> converting it to the grayscale
    image = Image.open(image_path).convert("L")
    image = np.array(image)

    #lowering the depth of brightness
    image = (image // (256 // levels)).astype(np.uint8)
    #computing parameters
    glcm = skimage.feature.graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    #averaging features across all distances and angles
    features = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        values = skimage.feature.graycoprops(glcm, prop)
        mean_value = values.mean()
        features[prop] = mean_value

    return features

def compute_glcm_features_to_csv(slc1_path, slc2_path, slc3_path, output_csv):
    header = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'category'] #header of csv file
    
    #saving features to the csv file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(header)
        
        for folder_path in [slc1_path, slc2_path, slc3_path]:
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(".jpg"):
                    path = os.path.join(folder_path, filename)
                    features = glcm_features(path)
                    category = os.path.basename(os.path.dirname(path))
                    row = [features[key] for key in header[0:-1]] + [category]
                    writer.writerow(row)

#exec the functions
slice_image_randomly("cw3/textures/gres/gres.jpg", "cw3/textures_samples/gres", 360, 50)
slice_image_randomly("cw3/textures/laminat/laminat.jpg", "cw3/textures_samples/laminat", 360, 50)
slice_image_randomly("cw3/textures/tynk/tynk.jpg", "cw3/textures_samples/tynk", 360, 50)
compute_glcm_features_to_csv("cw3/textures_samples/gres", "cw3/textures_samples/laminat", "cw3/textures_samples/tynk", "cw3/ftr.csv")

#START OF THE PROCESS OF LEARNING AND TESTING THE ALGORITHM #########################################
                                                                                                    #
#loading the data                                                                                   #
features = pandas.read_csv('cw3/ftr.csv', sep=';')                                                  #
data = np.array(features)                                                                           #
X = (data[:,0:-1]).astype('float64')                                                                #
Y = data[:,-1]                                                                                      #
                                                                                                    #
#scaling data for better accuracy                                                                   #
scaler = StandardScaler()                                                                           #
X = scaler.fit_transform(X)                                                                         #
                                                                                                    #
#process of learning                                                                                #
x_transform = PCA(n_components=3)                                                                   #
X_t = x_transform.fit_transform(X)                                                                  #
classifier = svm.SVC(gamma='scale')                                                                 #
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)                           #
classifier.fit(x_train, y_train)                                                                    #
y_pred = classifier.predict(x_test)                                                                 #
acc = accuracy_score(y_test, y_pred)                                                                #
                                                                                                    #
#formatting the test data in form of a matrix                                                       #
cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)                                   #
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)              #
                                                                                                    #
#displaying the data                                                                                #
print(f"\nAccuracy of the algorithm: {acc}\n")                                                      #
print(f"Confusion matrix:\n{cm}")                                                                   #
disp.plot()                                                                                         #
plt.show()                                                                                          #
#END OF THE PROCESS OF LEARNING AND TESTING THE ALGORITHM ###########################################