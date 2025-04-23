import numpy as np
from csv import writer


def generate_cylinder(radius, height, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(-height / 2, height / 2, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points = zip(x, y, z)
    return points

def generate_flat_horizontal(width, length, num_points, z_value=0.0):
    x = np.random.uniform(-width / 2, width / 2, num_points)
    y = np.random.uniform(-length / 2, length / 2, num_points)
    z = np.full(num_points, z_value)
    points = zip(x, y, z)
    return points


def generate_flat_vertical(width, height, num_points, x_value=0.0):
    y = np.random.uniform(-width/ 2, width / 2, num_points)
    z = np.random.uniform(-height / 2, height / 2, num_points)
    x = np.full(num_points, x_value)
    points = zip(x, y, z)
    return points


def save_to_xyz(filename, gen_points):
    with open(filename, 'w', newline='\n') as file:
        csvwriter = writer(file)
        for p in gen_points:
            csvwriter.writerow(p)

#generating cloud points for each of the surfaces
c_points = generate_cylinder(2.5, 5, 1000)
h_points = generate_flat_horizontal(5, 5, 1000)
v_points = generate_flat_vertical(5, 5, 1000)

#saving each set of cloud points to the .xyz files
# !!! downloaded source files should be put in 'cw1' folder
save_to_xyz("cw1/cylinder.xyz", c_points) 
save_to_xyz("cw1/horizontal.xyz", h_points)
save_to_xyz("cw1/vertical.xyz", v_points)
