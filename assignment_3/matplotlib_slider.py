import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

image_path = 'images/museum.jpg'
original_image = cv2.imread(image_path)
original_image = original_image[..., ::-1]
img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Create a sample image
# image_data = np.ones((100, 100, 3))
image_data = cv2.imread(image_path).astype(float)/255
image_data = image_data[..., ::-1]
img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Initial values for R, G, B channels
r_init, g_init, b_init = 1.0, 1.0, 1.0

# Apply initial RGB values to the image
#image_data[:, :, 0] *= r_init  # Red channel
#image_data[:, :, 1] *= g_init  # Green channel
#image_data[:, :, 2] *= b_init  # Blue channel

# Create the figure and image plot
fig, ax = plt.subplots()
img_plot = ax.imshow(image_data)
ax.axis('off')  # Turn off axis

# Create the position for sliders below the image
plt.subplots_adjust(bottom=0.3)

# Slider axes for R, G, B
ax_r = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor='lightgrey')
ax_g = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor='lightgrey')
ax_b = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgrey')

# Create sliders
slider_r = Slider(ax_r, 'Red', 0, 1, valinit=r_init)
slider_g = Slider(ax_g, 'Green', 0, 1, valinit=g_init)
slider_b = Slider(ax_b, 'Blue', 0, 1, valinit=b_init)

# Update function for sliders
def update(val):
    r = slider_r.val
    g = slider_g.val
    b = slider_b.val
    res = image_data.copy()
    res[:, :, 0] *= r
    res[:, :, 1] *= g
    res[:, :, 2] *= b
    img_plot.set_data((res*255).astype(np.uint8))
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_r.on_changed(update)
slider_g.on_changed(update)
slider_b.on_changed(update)

plt.show()
