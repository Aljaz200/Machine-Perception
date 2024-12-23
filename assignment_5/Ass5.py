#!/usr/bin/env python
# coding: utf-8

# ## Exercise 1: Disparity

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import a5_utils
import cv2

f = 0.0025
T = 0.12

pz = np.linspace(0.05, 10, 1000)

d = f * T / pz

plt.figure(figsize=(8, 5))
plt.plot(pz, d)
plt.title("Disperity vs Distance")
plt.xlabel("Distance (m)")
plt.ylabel("Disperity (m)")
plt.grid()
plt.show()


# #### What is the relation between the distance of the object (point p in Figure 1) to the camera and the disparity d? What happens to disparity when the object is close to the cameras and what when it is far away?
# 
# d = x1 - x2
# x1 / f = px / pz
# -x2 / f = (T - px) / pz
# 
# x1 = f * px / pz
# x2 = -f * (T - px) / pz
# 
# d = x1 - x2 = f * px / pz + f * (T - px) / pz = (f * px + f * T - f * px) / pz = f * T / pz
# 
# d = f * T / pz
# 
# Disparity d is inversely proportional to the distance of the object to the camera. When the object is close to the cameras, the disparity is large, and when the object is far away, the disparity is small.
# 

# In[2]:


def calculate_disparity_distance(f, T, x1, x2, pixel_width):
    d = (x1 - x2) * pixel_width
    pz = (f * T) / d # d = f * T / pz
    return pz, d


f = 0.0025
T = 0.12
pixel_width = 7.4e-6

x1_left = 550
x2_right1 = 300
x2_right2 = 540

pz1, d1 = calculate_disparity_distance(f, T, x1_left, x2_right1, pixel_width)
pz2, d2 = calculate_disparity_distance(f, T, x1_left, x2_right2, pixel_width)

print("Prvi primer:")
print(f"Disparity: {d1:.6f} m")
print(f"Razdalja objekta: {pz1:.2f} m")

print("\nDrugi primer:")
print(f"Disparity: {d2:.6f} m")
print(f"Razdalja objekta: {pz2:.2f} m")


# ## Exercise 2: Fundamental matrix, epipoles, epipolar lines

# In[3]:


F = np.array([
    [1, 0, 0],
    [0, 0.5, 0],
    [0, 0, -1]
])

points_left = [[0, 2, 1], [1, 0, 1]]

epipolar_lines = []
for point in points_left:
    line = F @ np.array(point)  # l' = F * x
    epipolar_lines.append(line)

for i, line in enumerate(epipolar_lines):
    print(f"Epipolarna èrta za toèko {points_left[i][:-1]}:")
    print(f"l' = {line[0]:.2f} * u' + {line[1]:.2f} * v' + {line[2]:.2f} -> [u', v', 1] * l' = 0")


# In[4]:


def fundamental_matrix(points):
    points_left = points[:, :2]
    points_right = points[:, 2:]

    normalized_left, T1 = a5_utils.normalize_points(points_left)
    normalized_right, T2 = a5_utils.normalize_points(points_right)

    A = []
    for (x1, y1), (x2, y2) in zip(normalized_left[:, :2], normalized_right[:, :2]):
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]) # sistem linearnih enaèb
    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    F_hat = Vt[-1].reshape(3, 3) # transponiramo zadnji lastni vektor v matriko 3x3

    U, S, Vt = np.linalg.svd(F_hat) # ponovimo razcep na lastne vrednosti in lastne vektorje
    S[-1] = 0 # zadnjo lastno vrednost postavimo na 0
    F_hat = U @ np.diag(S) @ Vt # sestavimo nazaj matriko F

    F = T1.T @ F_hat @ T2
    return F.T


points = np.loadtxt('data/epipolar/house_points.txt')
#F_correct = np.loadtxt('data/epipolar/house_fundamental.txt')

F = fundamental_matrix(points)
print("F:")
print(F)
#print("\nF correct:")
#print(F_correct)

points_left = points[:, :2]
points_right = points[:, 2:]

image_left = cv2.imread('data/epipolar/house1.jpg', cv2.IMREAD_GRAYSCALE)
image_right = cv2.imread('data/epipolar/house2.jpg', cv2.IMREAD_GRAYSCALE)
h, w = image_left.shape

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.imshow(image_left, cmap='gray')
plt.title('Epipolarne èrte v levi sliki')

for point1, point2 in zip(points_left, points_right):
    line = F.T @ np.array([point2[0], point2[1], 1])
    a5_utils.draw_epiline(line, h, w)
    plt.scatter(point1[0], point1[1], color='red', s=20)

plt.subplot(1, 2, 2)
plt.imshow(image_right, cmap='gray')
plt.title('Epipolarne èrte v desni sliki')

for point1, point2 in zip(points_left, points_right):
    line = F @ np.array([point1[0], point1[1], 1])
    a5_utils.draw_epiline(line, h, w)
    plt.scatter(point2[0], point2[1], color='red', s=20)

plt.show()


# In[5]:


def reprojection_error(F, p1, p2):
    p1_hom = np.append(p1, 1)
    p2_hom = np.append(p2, 1)

    # distance(ax + by + c = 0, (x0, y0)) = |a * x0 + b * y0 + c| / sqrt(a^2 + b^2) -> razdalja toèke od premice
    
    l2 = F @ p1_hom
    a, b, c = l2
    distance_2 = abs(a * p2[0] + b * p2[1] + c) / np.sqrt(a**2 + b**2)
    
    l1 = F.T @ p2_hom
    a, b, c = l1
    distance_1 = abs(a * p1[0] + b * p1[1] + c) / np.sqrt(a**2 + b**2)
    
    avg_error = (distance_1 + distance_2) / 2
    return avg_error

points = np.loadtxt('data/epipolar/house_points.txt')
F = fundamental_matrix(points)

p1 = np.array([85, 233])
p2 = np.array([67, 219])

error = reprojection_error(F, p1, p2)
print(f"Reprojection error: {error:.4f} pixels")

errors = []
for i in range(len(points)):
    p1 = points[i, :2]
    p2 = points[i, 2:]
    error = reprojection_error(F, p1, p2)
    errors.append(error)

average_error = np.average(errors)
print(f"Average reprojection error: {average_error:.4f} pixels")


# ## Exercise 3: Triangulation

# In[6]:


def triangulate(points, P1, P2):
    points_3D = []
    
    for x1, y1, x2, y2 in points:
        A = np.zeros((4, 4))
        A[0] = x1 * P1[2] - P1[0]
        A[1] = y1 * P1[2] - P1[1]
        A[2] = x2 * P2[2] - P2[0]
        A[3] = y2 * P2[2] - P2[1]

        # rešimo sistem linearnih enaèb
        
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[-1] # normalizacija
        points_3D.append(X[:3])
    
    return np.array(points_3D)

points = np.loadtxt('data/epipolar/house_points.txt')
P1 = np.loadtxt('data/epipolar/house1_camera.txt')
P2 = np.loadtxt('data/epipolar/house2_camera.txt')

points_left = points[:, :2]
points_right = points[:, 2:]

points_3D = triangulate(points, P1, P2)

T = np.array([[-1, 0, 0], 
              [0, 0, -1], 
              [0, 1, 0]])
points_3d_transformed = points_3D @ T.T # transformacija za prikaz v kooridnatnem sistemu


image_left = cv2.imread('data/epipolar/house1.jpg', cv2.IMREAD_GRAYSCALE)
image_right = cv2.imread('data/epipolar/house2.jpg', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_left, cmap='gray')
plt.scatter(points_left[:, 0], points_left[:, 1], color='red', s=20)

for i, point1 in enumerate(points_left):
    plt.text(point1[0], point1[1], f'{i}', color='black')


plt.subplot(1, 2, 2)
plt.imshow(image_right, cmap='gray')
plt.scatter(points_right[:, 0], points_right[:, 1], color='red', s=20)

for i, point2 in enumerate(points_right):
    plt.text(point2[0], point2[1], f'{i}', color='black')

plt.show()

fig = plt.figure(figsize=(10, 8))
pos = fig.add_subplot(111, projection='3d')
pos.scatter(points_3d_transformed[:, 0], points_3d_transformed[:, 1], points_3d_transformed[:, 2], color='red', marker='o', s=20)
for i, point in enumerate(points_3d_transformed):
    pos.text(point[0], point[1], point[2], f'{i}', color='black')

plt.title("3D Reconstruction points")
plt.show()

