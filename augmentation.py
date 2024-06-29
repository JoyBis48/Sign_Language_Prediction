import numpy as np
from tqdm import tqdm

# Rotating augmentations
def rotate(data, rotation_matrix):
    frames, landmarks, _ = data.shape
    center = np.array([0.5, 0.5, 0])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data_flat = data.reshape(-1, 3)
    non_zero_data = data_flat[non_zero[:, 0], :]
    non_zero_data -= center
    non_zero_data = np.dot(non_zero_data, rotation_matrix.T)
    non_zero_data += center
    data_flat[non_zero[:, 0], :] = non_zero_data
    data = data_flat.reshape(frames, landmarks, 3)
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data

def rotate_z(data):
    angle = np.random.uniform(-30, 30)
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return rotate(data, rotation_matrix)

def rotate_y(data):
    angle = np.random.uniform(-30, 30)
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return rotate(data, rotation_matrix)

def rotate_x(data):
    angle = np.random.uniform(-30, 30)
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return rotate(data, rotation_matrix)

# Other Augmentations

def zoom(data):
    factor = np.random.uniform(0.8, 1.2)
    center = np.array([0.5, 0.5])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], :2] -= center
    data[non_zero[:, 0], non_zero[:, 1], :2] *= factor
    data[non_zero[:, 0], non_zero[:, 1], :2] += center
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data

def shift(data):
    shift_values = np.random.uniform(-0.2, 0.2, size=2)
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], :2] += shift_values
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data

def mask(data):
    frames, landmarks, _ = data.shape
    num_hands = int(0.3 * 42)
    num_rest = int(0.6 * (landmarks - 42))
    indices = np.random.choice(landmarks, num_hands + num_rest, replace=False)
    data[:, indices] = 0
    return data

def hflip(data):
    data[:, :, 0] = 1 - data[:, :, 0]
    return data

def speedup(data):
    return data[::2]

# defining Function for applying augmentations

def apply_augmentations(data):
    aug_functions = [rotate_x, rotate_y, rotate_z, zoom, shift, mask, hflip, speedup]
    np.random.shuffle(aug_functions)
    for fun in aug_functions:
        if np.random.rand() < 0.5:
            data = fun(data)
    return data

def augment(X, Y, num=None):
    X_aug, Y_aug = [], []
    for i in tqdm(range(len(Y)), ncols=100):
        num_aug = np.random.choice([1, 2, 3]) if num is None else num
        for _ in range(num_aug):
            X_aug.append(apply_augmentations(X[i].copy()))
            Y_aug.append(Y[i])
    return X_aug, Y_aug