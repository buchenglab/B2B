
"""
Created on Tue Oct 13 17:28:37 2026

@author: Guangrui Ding
"""

import numpy as np
from skimage import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil

def random_pixel_shuffle_3d(stack, xy_range=1, z_range=1):
    """
    Randomly replace each voxel in a 3D stack with another voxel within a local neighborhood.

    Parameters:
    - stack: np.ndarray, shape (H, W, D), the original 3D data stack
    - xy_range: int, defines neighborhood radius in the XY-plane
    - z_range: int, defines neighborhood radius in the Z-axis

    Returns:
    - new_stack: np.ndarray, same shape as input, with shuffled voxels
    """
    H, W, D = stack.shape
    new_stack = np.zeros_like(stack)

    for x in range(W):
        for y in range(D):
            for z in range(H):
                # Define random offset within legal range
                dx = np.random.randint(-xy_range, xy_range + 1)
                dy = np.random.randint(-xy_range, xy_range + 1)
                dz = np.random.randint(-z_range, z_range + 1)

                # Clip the coordinate to stay within bounds
                x_new = np.clip(x + dx, 0, W - 1)
                y_new = np.clip(y + dy, 0, D - 1)
                z_new = np.clip(z + dz, 0, H - 1)
                # print(x_new, y_new,z_new)
                # Assign random neighbor value
                new_stack[z, x, y] = stack[z_new, x_new, y_new]

        print(x)

    return new_stack


def fast_random_pixel_shuffle_zxy(stack, x_range=1, y_range=1, z_range=1):
    """
    Randomly replaces each voxel in a (Z, X, Y) stack with another voxel
    within a local neighborhood defined by z_range, x_range, y_range.

    Parameters:
        stack: np.ndarray of shape (Z, X, Y)
        x_range, y_range, z_range: int, half-window sizes in respective axes

    Returns:
        new_stack: np.ndarray of same shape with randomly permuted pixels
    """
    Z, X, Y = stack.shape

    # Generate full index grid
    zz, xx, yy = np.meshgrid(
        np.arange(Z), np.arange(X), np.arange(Y), indexing='ij'
    )

    # Sample random offsets within the given neighborhood range
    dz = np.random.randint(-z_range, z_range + 1, size=stack.shape)
    dx = np.random.randint(-x_range, x_range + 1, size=stack.shape)
    dy = np.random.randint(-y_range, y_range + 1, size=stack.shape)

    # Compute new randomized coordinates, clipped to valid range
    zn = np.clip(zz + dz, 0, Z - 1)
    xn = np.clip(xx + dx, 0, X - 1)
    yn = np.clip(yy + dy, 0, Y - 1)

    # Apply permutation
    new_stack = stack[zn, xn, yn]
    return new_stack

def fast_random_pixel_shuffle_zxy_minmax(stack,
                                    x_min=1, x_max=2,
                                    y_min=1, y_max=2,
                                    z_min=1, z_max=1):
    """
    Randomly replaces each voxel in (Z, X, Y) stack with another voxel
    within a specified *minimum-to-maximum* offset neighborhood.

    Parameters:
        stack: np.ndarray of shape (Z, X, Y)
        *_min, *_max: range limits for random displacement (excluding center)

    Returns:
        new_stack: shuffled version of stack
    """
    Z, X, Y = stack.shape

    # Generate full index grid
    zz, xx, yy = np.meshgrid(np.arange(Z), np.arange(X), np.arange(Y), indexing='ij')

    def sample_offset(shape, min_val, max_val):
        """
        Sample from [-max, -min] ∪ [min, max]
        """
        delta = np.random.randint(min_val, max_val + 1, size=shape)
        sign = np.random.choice([-1, 1], size=shape)
        return delta * sign

    # Random displacements with min/max range
    dz = sample_offset(stack.shape, z_min, z_max)
    dx = sample_offset(stack.shape, x_min, x_max)
    dy = sample_offset(stack.shape, y_min, y_max)

    # New coordinates with clipping to valid bounds
    zn = np.clip(zz + dz, 0, Z - 1)
    xn = np.clip(xx + dx, 0, X - 1)
    yn = np.clip(yy + dy, 0, Y - 1)

    # Apply permutation
    new_stack = stack[zn, xn, yn]
    return new_stack

def img_save(img, Name, Target_folder, index, count):
    save_path = Target_folder + '/' + Name + '_' + str(index) + '_' + str(count) + '.tif'
    io.imsave(save_path, img)

# Change your folder name here
folder_path = "./data/20260507_SWIP_Spheroid"
target_path_1 = "./data/Input"
target_path_2 = "./data/Target"

if os.path.exists(target_path_1):
    shutil.rmtree(target_path_1)

if os.path.exists(target_path_2):
    shutil.rmtree(target_path_2)

if not os.path.exists(target_path_1):
    os.mkdir(target_path_1)
if not os.path.exists(target_path_2):
    os.mkdir(target_path_2)

# customize the parameter based on your feature size
x_range = 20
y_range = x_range
z_range = 1

x_range_min = 0
y_range_min = x_range_min
z_range_min = 0

num_repeat = 1

img_name = os.listdir(folder_path)
for index, name in enumerate(img_name):
    img_path = folder_path + '/' + name
    img = io.imread(img_path)

    for time in range(num_repeat):
        # img_shuffle_1 = fast_random_pixel_shuffle_zxy(img, x_range,y_range,z_range)
        # img_shuffle_2 = fast_random_pixel_shuffle_zxy(img, x_range,y_range,z_range)

        # img_shuffle_1 = fast_random_pixel_shuffle_zxy_minmax(img, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)
        img_shuffle_1 = img
        print(img.shape)
        # img_shuffle_1 = img - fast_random_pixel_shuffle_zxy_minmax(img, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)

        img_shuffle_2 = fast_random_pixel_shuffle_zxy_minmax(img, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)
        # img_shuffle_2 = img - fast_random_pixel_shuffle_zxy_minmax(img, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)
        # img_shuffle_2 = img
        img_save(img_shuffle_1, name, target_path_1, 0, time)
        img_save(img_shuffle_2, name, target_path_2, 0, time)

        # img1 = np.flip(img, 1)
        # img_shuffle_1 = img1
        # # img_shuffle_1 = fast_random_pixel_shuffle_zxy_minmax(img1, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)
        # img_shuffle_2 = fast_random_pixel_shuffle_zxy_minmax(img1, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)
        # # # # img_shuffle_2 = img
        # img_save(img_shuffle_1, name, target_path_1, 1, time)
        # img_save(img_shuffle_2, name, target_path_2, 1, time)
        # #
        # #
        # img2 = np.flip(img, 2)
        # img_shuffle_1 = img2
        # # img_shuffle_1 = fast_random_pixel_shuffle_zxy_minmax(img2, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)
        # img_shuffle_2 = fast_random_pixel_shuffle_zxy_minmax(img2, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)
        # # # # img_shuffle_2 = img
        # img_save(img_shuffle_1, name, target_path_1, 2, time)
        # img_save(img_shuffle_2, name, target_path_2, 2, time)
        # #
        # #
        # img3 = np.flip(img1,2)
        # img_shuffle_1 = img3
        # # img_shuffle_1 = fast_random_pixel_shuffle_zxy_minmax(img, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)+img
        # img_shuffle_2 = fast_random_pixel_shuffle_zxy_minmax(img3, x_min=x_range_min, x_max=x_range, y_min=y_range_min, y_max=y_range,z_min=z_range_min, z_max=z_range)
        # # # img_shuffle_2 = img
        # img_save(img_shuffle_1, name, target_path_1, 3, time)
        # img_save(img_shuffle_2, name, target_path_2, 3, time)



