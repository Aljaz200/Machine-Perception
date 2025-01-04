#!/usr/bin/env python
# coding: utf-8

# ## Exercise 1: Direct PCA method

# In[43]:


import numpy as np
import matplotlib.pyplot as plt
import a6_utils
import cv2
import os
import time

def compute_mean(data):
    return np.mean(data, axis=0)

def center_data(data, mean):
    return data - mean

def compute_covariance_matrix(data):
    return np.cov(data, rowvar=False)

def perform_svd(cov_matrix):
    U, S, VT = np.linalg.svd(cov_matrix)
    return U, S, VT

def project_to_pca_space(data, mean, eigenvectors):
    centered_data = center_data(data, mean)
    return centered_data @ eigenvectors.T

def reconstruct_from_pca_space(projected_data, mean, eigenvectors):
    return projected_data @ eigenvectors + mean

def PCA(data):
    mean = compute_mean(data)
    centered_data = center_data(data, mean)
    cov_matrix = compute_covariance_matrix(centered_data)
    eigenvectors, eigenvalues, _ = perform_svd(cov_matrix)
    return project_to_pca_space(data, mean, eigenvectors), mean, eigenvectors, eigenvalues



# In[2]:




# In[3]:

# ### What do you notice about the relationship between the eigenvectors and the data? What happens to the eigenvectors if you change the data or add more points?
# 
# The eigenvectors are the directions of the data that have the most variance. If we change the data or add more points, the eigenvectors will change. The eigenvectors will always be orthogonal to each other, but the directions of the eigenvectors will change.

# In[4]:



# In[5]:




# ### What happens to the reconstructed points? Where is the data projected to?
# 
# The reconstructed points are projected to the eigenvectors. The data is projected to the eigenvectors.

# In[6]:



# ## Exercise 2: The dual PCA method

# In[7]:


def dual_pca(data):
    mean = compute_mean(data)
    centered_data = center_data(data, mean)
    dual_cov_matrix = np.dot(centered_data.T, centered_data) / (len(data) - 1)
    eigenvectors, eigenvalues, _  = np.linalg.svd(dual_cov_matrix)
    U_original = np.dot(centered_data, eigenvectors) * np.sqrt(1 / (eigenvalues * (len(data) - 1)))

    U_original[:, 2:] = 0

    return project_to_pca_space(data, mean, U_original), mean, eigenvectors, eigenvalues, U_original



# In[8]:



# ## Exercise 3: Image decomposition examples

# In[26]:


def prepare_image_matrix(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        img_vector = np.reshape(np.array(img), -1)
        images.append(img_vector)
    return np.array(images).T

image_folder1 = "data/faces/1"
image_matrix = prepare_image_matrix(image_folder1)
print("Shape of the image matrix:", image_matrix.shape)


# In[27]:


def dual_pca_image(image_matrix):
    mean_images = compute_mean(image_matrix.T)
    centered_images = center_data(image_matrix.T, mean_images)

    dual_cov_matrix = np.dot(centered_images, centered_images.T) / (centered_images.shape[0] - 1)
    U, S, _ = perform_svd(dual_cov_matrix)
    S = S + 1e-15
    eigenvectors = np.dot(centered_images.T, U) / np.sqrt(S * (centered_images.shape[0] - 1))

    return eigenvectors, S, mean_images



# ### What do the resulting images represent (both numerically and in the context of faces)?
# 
# The resulting images represent the eigenfaces. Eigenvectors define axes in the PCA space, and the eigenfaces are the images that correspond to these axes. First eigenvector represents the direction of the most variance in the data, and the second eigenvector represents the direction of the second most variance in the data, and so on. 
# The eigenfaces are the principal components of the faces. Each face can be represented as a linear combination of the eigenfaces.
# 
# ### What is the difference? How many pixels are changed by the first operation and how many by the second?
# 
# In the original space, the change only effects one pixel. In the PCA space, the change effects all the pixels, because PCA linearly combines the eigenvectors to reconstruct the image.
# The change in the original space is only one pixel, but the change in the PCA space is global, because the PCA components define face features as a whole.

# In[28]:




# ### Display the resulting vectors together on one figure. What do you notice?
# 
# Reconstructed images with more components are more similar to the original image. The first component is the most important component, and the second component is the second most important component, and so on. The more components we use, the more similar the reconstructed image is to the original image.
# With small number of components the details are lost, but the general shape of the face is preserved. With more components the details are preserved, and the reconstructed image is more similar to the original image.

# In[67]:


def draw_rectangle(image, x, y, w, h, label, color=(1, 0, 0), thickness=2):
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=thickness))
    plt.text(x, y, label, color=color)
    plt.axis('off')
    plt.show()


def recognize_faces(pca, mean_face, resolution=(64, 64), threshold=1500):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for _ in range(3):
        time.sleep(2)
        print("Capturing camera image...")
        time.sleep(1)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, resolution).flatten()

            projected_face = project_to_pca_space(face_resized, mean_face, pca)
            reconstructed_face = reconstruct_from_pca_space(projected_face, mean_face, pca)

            error = np.linalg.norm(face_resized - reconstructed_face)
            color = (0, 1, 0) if error < threshold else (1, 0, 0)
            label = "Recognized" if error < threshold else "Unknown"

            print(f"Error: {error:.2f}")
            draw_rectangle(frame, x, y, w, h, label, color=color)

    cap.release()
    cv2.destroyAllWindows()

image_folder1 = "data/faces/Aljaz"
image_matrix = prepare_image_matrix(image_folder1)
eigenvectors, eigenvalues, mean_image = dual_pca_image(image_matrix)

recognize_faces(eigenvectors.T, mean_image, resolution=(100, 80), threshold=6000)

