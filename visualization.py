import numpy as np
from numpy import random
import trimesh
from einops import rearrange

## Lets simulate some fishes
# We're using cm for all unit hereby
# Note: x = horizontal of camera frame, y = vertical of camera frame, z = along camera frame, assume the camera frame is align with the global frame
# Define vertices and triangles for a single fish
# actual size of the fish: 17cm long * 8 cm height

def gimmi_a_fish(mean_scale = 1):
    fish_vertices = np.array([[0, 0, 0], [6, 4, 0], [6, -4, 0],  [15, 0, 0], [17,3,0], [17, -3, 0]], dtype=float)
    fish_faces = np.array([[0, 2, 1], [1, 3, 2], [3, 5, 4], [0, 1, 2], [1, 2, 3], [3, 4, 5]])

    # now we randomly scale and rotate (about the y axis) the fish
    # random variables
    scale = random.normal()*0.5 + mean_scale   #normal distribution 
    theta = 2*np.pi*random.rand()
    
    # scaling
    fish_vertices *= scale

    # rotation
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.array([[c, 0, -s],[0, 1, 0], [s, 0, c]])
    fish_vertices = rearrange(fish_vertices, 'a b -> b a')
    fish_vertices = rotation@fish_vertices
    fish_vertices = rearrange(fish_vertices, 'a b -> b a')

    # translation (in a 3m x 3m x 3m tank)
    translation = random.rand(3)*300
    translation = np.tile(translation,(len(fish_vertices),1))
    fish_vertices = fish_vertices + translation
    return fish_vertices, fish_faces


def gimmi_many_fish(numb, mean_scale = 1):
    a, b = gimmi_a_fish(mean_scale)
    for i in range(numb-1):
        vertices, faces = gimmi_a_fish(mean_scale)
        faces = faces + 6*(i+1)
        a = np.vstack((a, vertices))
        b = np.vstack((b, faces))
    return a, b




def main():
    # Create the mesh
    a, b = gimmi_many_fish(100)
    mesh = trimesh.Trimesh(vertices=a, faces=b, process = False)

    # Draw field of view
    ray_visualizations = []

    # Ray origins and directions
    ray_origins = np.array([
        [150, 150, -20],
        [150, 150, -20],
        [150, 150, -20],
        [150, 150, -20]
    ])
    ray_directions = np.array([
        [-0.960,-0.540, 3.5],
        [-0.960,0.540, 3.5],
        [0.960,-0.540, 3.5],
        [0.960, 0.540, 3.5]
    ])

    # Parameters for the rays
    ray_length = 80
    cylinder_radius = 0.3  # Adjust the radius for desired thickness

    # Create cylinders for each ray
    for origin, direction in zip(ray_origins, ray_directions):
        end_point = origin + ray_length * np.array(direction)
        # Define the segment as two points
        segment = np.array([origin, end_point])
        # Create a cylinder along the segment
        cylinder = trimesh.creation.cylinder(radius=cylinder_radius, segment=segment)
        ray_visualizations.append(cylinder)

    # Create the scene with the mesh and ray visualizations
    scene = trimesh.Scene([mesh] + ray_visualizations)
    # Display the scene
    scene.show()
    ##############################################################

    # Now, we can try to visualize fishes projected onto the image
    # We assumed the camera has a focallength of 3.5cm, and did not rotate in the 
    # Assuming the camera can be calibrated so we don't have to worry about distortion
    extrinsic = np.array([[1, 0, 0, -150], [0, 1, 0, -150], [0, 0, 1, 20]])
    intrinsic = np.array([[3.5, 0, 0], [0, 3.5, 0], [0, 0, 1]])
    projection = intrinsic@extrinsic
    # adding a row of ones to a so that we have homogenous coordinates of vertices of fishes
    temp = np.ones((len(a), 1))
    a = rearrange(a, 'a b -> b a')
    temp = rearrange(temp, 'a b -> b a')
    a = np.vstack((a, temp))
    # Apply projection matrix, and divided by homogenous constant, to make the "third value" 1
    projected = projection@a
    projected = rearrange(projected, 'a b -> b a')
    projected[:,0] = np.divide(projected[:,0], projected[:,2])
    projected[:,1] = np.divide(projected[:,1], projected[:,2])
    # don't really need to set this value actually, but for sake of completeness, choosign z axis as -20+3.5=-16.5 will be the actual location of the sensor
    projected[:,2] = -16.6  
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=projected, faces=b, process = False)

    # Draw field of view
    ray_visualizations = []

    # Ray origins and directions
    ray_origins = np.array([
        [0.96, 0.54, -16.5],
        [-0.96, -0.54, -16.5],
        [0.96, -0.54, -16.5],
        [-0.96, 0.54, -16.5]
    ])
    ray_directions = np.array([
        [0,-1, 0],
        [0, 1, 0],
        [-1,0 , 0],
        [1, 0, 0]
    ])


    # Parameters for the rays
    ray_length = [1.08, 1.08, 1.92, 1.92]
    cylinder_radius = 0.005  # Adjust the radius for desired thickness

    # Create cylinders for each ray
    idx = 0
    for origin, direction in zip(ray_origins, ray_directions):
        end_point = origin + ray_length[idx] * np.array(direction)
        # Define the segment as two points
        segment = np.array([origin, end_point])
        # Create a cylinder along the segment
        cylinder = trimesh.creation.cylinder(radius=cylinder_radius, segment=segment)
        ray_visualizations.append(cylinder)
        idx += 1

    # Create the scene with the mesh and ray visualizations
    scene = trimesh.Scene([mesh] + ray_visualizations)

    # scene = trimesh.Scene([mesh] )
    # Display the scene
    scene.show()


if __name__ == "__main__":
    main()

