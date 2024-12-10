import cv2
import numpy as np

window_name = 'Image Processing - Blur and Threshold'


def morph_slider(val, im):


	val = cv2.getTrackbarPos('Threshold', window_name)	
	kernel_size = cv2.getTrackbarPos('Morphology', window_name)

	#print(f"{kernel_size=}")

	gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	thr = (gray_image<val).astype(np.uint8)

	# print(np.unique(thr))


	#kernel = np.ones((kernel_size,kernel_size), np.uint8)

	# print(kernel)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))

	res = cv2.dilate(thr, kernel, iterations = 1)

	# print(np.unique(res))


	res = (res*255).astype(np.uint8)

	# print(np.unique(res))

	cv2.imshow(window_name, res)


	# print(val)
	# print(im.shape)

def threshold_slider(val, im):

	#print(val)
	#print(im.shape)

	gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


	res = ((gray_image<val)*255).astype(np.uint8)

	cv2.imshow(window_name, res)
	
def update_image(val):
    """Callback to update the image based on trackbar values."""
    # Start with the original image
    processed_image = original_image.copy()

    # Apply blur first
    final_image = apply_blur(processed_image)

    # Apply thresholding to the blurred image
    #final_image = apply_threshold(blurred_image)

    # Show the processed image
    cv2.imshow(window_name, final_image)

# Load your image
image_path = 'images/museum.jpg'
original_image = cv2.imread(image_path)

# Create a window to display the image and resize it to make room for trackbars
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(window_name, 600, 400)  # Resize the window for better visibility


cv2.createTrackbar('Threshold', window_name, 1, 255, lambda x: threshold_slider(x, original_image))
cv2.createTrackbar('Morphology', window_name, 1, 50, lambda x: morph_slider(x, original_image))

cv2.setTrackbarMin('Morphology', window_name, 1) 

cv2.imshow(window_name, original_image)
# cv2.imshow("morph", original_image)

initial_threshold = 127

threshold_slider(initial_threshold, original_image)
cv2.setTrackbarPos('Threshold', window_name, initial_threshold)
# cv2.setTrackbarPos('Morphology', window_name, 1)

# Wait for user interaction
cv2.waitKey(0)
cv2.destroyAllWindows()


# Define the kernel shapes and morphological operations
kernel_shapes = {
    "Rect": cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross": cv2.MORPH_CROSS
}

morph_operations = {
    "Erode": cv2.MORPH_ERODE,
    "Dilate": cv2.MORPH_DILATE,
    "Open": cv2.MORPH_OPEN,
    "Close": cv2.MORPH_CLOSE,
    "Gradient": cv2.MORPH_GRADIENT,
    "Top Hat": cv2.MORPH_TOPHAT,
    "Black Hat": cv2.MORPH_BLACKHAT
}

output_modes = ["Binary", "Grayscale", "Color"]

# Convert dictionary keys to lists for trackbar index mapping
kernel_shape_names = list(kernel_shapes.keys())
morph_operation_names = list(morph_operations.keys())

# Initialize trackbar values
selected_kernel_shape = 0  # Default: Rect
selected_morph_operation = 0  # Default: Erode
selected_output_mode = 1  # Default: Grayscale
kernel_size = 5  # Initial kernel size

# Load a user-defined image (replace with your image path)
image_path = 'images/museum.jpg'
original_image = cv2.imread(image_path)

if original_image is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale and binary for initial setup
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

# Callback functions for trackbars
def on_kernel_shape_trackbar(val):
    global selected_kernel_shape
    selected_kernel_shape = val
    apply_morphology()

def on_morph_operation_trackbar(val):
    global selected_morph_operation
    selected_morph_operation = val
    apply_morphology()

def on_kernel_size_trackbar(val):
    global kernel_size
    kernel_size = max(1, val * 2 + 1)  # Ensure kernel size is odd
    apply_morphology()

def on_output_mode_trackbar(val):
    global selected_output_mode
    selected_output_mode = val
    apply_morphology()

# Function to draw text with black border around white text for better visibility
def draw_text(image, text, position):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Black border
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # White text

# Function to apply the selected morphology operation to the image
def apply_morphology():
    # Get the selected kernel shape and operation
    kernel_shape = kernel_shapes[kernel_shape_names[selected_kernel_shape]]
    morph_operation = morph_operations[morph_operation_names[selected_morph_operation]]
    
    # Create the kernel with the selected shape and size
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    
    # Select the appropriate base image and apply morphology
    if selected_output_mode == 0:  # Binary
        base_image = binary_image
    elif selected_output_mode == 1:  # Grayscale
        base_image = gray_image
    else:  # Color
        base_image = original_image
    
    # Apply the morphology operation
    if selected_output_mode == 2:  # Color mode requires separate processing for each channel
        morphed_image = cv2.merge([cv2.morphologyEx(base_image[:, :, i], morph_operation, kernel) for i in range(3)])
    else:
        morphed_image = cv2.morphologyEx(base_image, morph_operation, kernel)
    
    # Display the morphed image with current selections
    display_img = np.copy(morphed_image)
    draw_text(display_img, f"Kernel Shape: {kernel_shape_names[selected_kernel_shape]}", (10, 20))
    draw_text(display_img, f"Operation: {morph_operation_names[selected_morph_operation]}", (10, 40))
    draw_text(display_img, f"Kernel Size: {kernel_size}", (10, 60))
    draw_text(display_img, f"Output Mode: {output_modes[selected_output_mode]}", (10, 80))
    
    cv2.imshow("Morphology Application", display_img)

# Create a window and trackbars with abbreviated names
cv2.namedWindow("Morphology Application", cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar("K Shape", "Morphology Application", 0, len(kernel_shapes) - 1, on_kernel_shape_trackbar)
cv2.createTrackbar("Op", "Morphology Application", 0, len(morph_operations) - 1, on_morph_operation_trackbar)
cv2.createTrackbar("K Size", "Morphology Application", 2, 20, on_kernel_size_trackbar)  # Max kernel size of 41 (2*20 + 1)
cv2.createTrackbar("Out Mode", "Morphology Application", 0, len(output_modes) - 1, on_output_mode_trackbar)

# Initial application of morphology
apply_morphology()

# Wait for the user to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
