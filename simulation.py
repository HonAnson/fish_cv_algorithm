import numpy as np
from numpy import random
import trimesh
from einops import rearrange
from visualization import gimmi_a_fish, gimmi_many_fish



# E(distance | fish is in field of view)

# calculate area of fish in image frame
def get_area(vertices):
    x_coord = vertices[:,0]
    y_coord = vertices[:,1]
    x_shifted = np.roll(x_coord, -1)
    y_shifted = np.roll(y_coord, -1)
    a = np.inner(x_coord, y_shifted)
    b = np.inner(x_shifted, y_coord)
    return abs((0.5)*(a-b))

# return the length of fish in image frame
def get_length(vertices):
    tail = (vertices[4,:] + vertices[5,:]) / 2
    head = vertices[0,:]
    return ((head[0] - tail[0])**2 + (head[1] - tail[1])**2)**0.5


# function for helping to calculate the corrsponding length and size of fish given average length and size of fish in image
def gimmi_a_fish_angled(theta, E_distance, scale):
    fish_vertices = np.array([[0, 0, 0], [6, 4, 0], [6, -4, 0],  [15, 0, 0], [17,3,0], [17, -3, 0]], dtype=float)

    # scaling
    fish_vertices[:,0:2] = fish_vertices[:,0:2]*scale

    # rotation
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.array([[c, 0, -s],[0, 1, 0], [s, 0, c]])
    fish_vertices = rearrange(fish_vertices, 'a b -> b a')
    fish_vertices = rotation@fish_vertices
    fish_vertices = rearrange(fish_vertices, 'a b -> b a')

    # translation
    translation = np.array([150, 150, E_distance])
    translation = np.tile(translation,(len(fish_vertices),1))
    fish_vertices = fish_vertices + translation
    return fish_vertices




def main():
    # First, note volume of the field of view:
    S1 = 1.08*1.92/(3.5/20)**2
    S2 = 1.08*1.92/(3.5/320)**2
    h = 300
    V = (S1 + S2 + (S1*S2)**0.5)*(h/3)
    E_distance = (1/V) * (1.08*1.92)/(3.5*3.5) * 0.25 * (320**4 - 20**3) # Calculation of expected value integral

    # expected size from rotation
    # We have to make a chart
    # for each scale, we rotate the fish 360 degree at different at expected distance, then record the result (since it is challenging to calculate the expected size and shape in closed form, this is left as future work of the algorithm I'm designing)
    all_area = []
    all_length = []
    scales = np.arange(0.5, 1.6, 0.1) # scale represents the size of fish
    for scale in scales:
        total_area = 0
        total_length = 0
        for i in range(360):
            theta = 2* np.pi * (i / 360)       # rotating fish from 0 to
            vertices = gimmi_a_fish_angled(theta, E_distance, scale)
            total_area += get_area(vertices[0:4,:])
            total_area += get_area(vertices[3:6,:])
            total_length += get_length(vertices)

        all_area.append(total_area / 360)
        all_length.append(total_length/360)



    ### Now we run our simulation


if __name__ == "__main__":
    main()