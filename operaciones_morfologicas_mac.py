import tkinter as tk
from tkinter import Scale
import numpy as np
from PIL import Image, ImageTk
import cv2

def update_hsv(event=None):
    # Get HSV values from sliders
    h_min, s_min, v_min = h_min_scale.get(), s_min_scale.get(), v_min_scale.get()
    h_max, s_max, v_max = h_max_scale.get(), s_max_scale.get(), v_max_scale.get()
    
    # Define HSV mask limits
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    # Convert image to HSV and apply mask
    hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    hsv_result = cv2.bitwise_and(img, img, mask=mask)
    
    # Update the HSV result display
    hsv_result_rgb = cv2.cvtColor(hsv_result, cv2.COLOR_BGR2RGB)
    hsv_photo = ImageTk.PhotoImage(image=Image.fromarray(hsv_result_rgb))
    hsv_canvas.img = hsv_photo
    hsv_canvas.create_image(0, 0, anchor=tk.NW, image=hsv_photo)
    
    # Update morphological result
    update_morph(mask)

def update_morph(mask):
    # Get morphological operation values
    erode_iterations = erode_iter_scale.get()
    dilate_iterations = dilate_iter_scale.get()
    erode_kernel_size = erode_kernel_scale.get()
    dilate_kernel_size = dilate_kernel_scale.get()
    
    # Create kernels for erosion and dilation
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    
    # Apply erosion and dilation on the mask
    eroded_mask = cv2.erode(mask, erode_kernel, iterations=erode_iterations)
    dilated_mask = cv2.dilate(eroded_mask, dilate_kernel, iterations=dilate_iterations)
    
    # Combine original image with the dilated mask
    morph_result = cv2.bitwise_and(img, img, mask=dilated_mask)
    morph_result_rgb = cv2.cvtColor(morph_result, cv2.COLOR_BGR2RGB)
    
    # Update the morphological result display
    morph_photo = ImageTk.PhotoImage(image=Image.fromarray(morph_result_rgb))
    morph_canvas.img = morph_photo
    morph_canvas.create_image(0, 0, anchor=tk.NW, image=morph_photo)

# Load image
img = cv2.imread("imagen.png")
img = cv2.resize(img, (480, 360))

# Create main HSV control window
hsv_window = tk.Tk()
hsv_window.title("HSV Color Picker")

# Default HSV values
h_min_default, s_min_default, v_min_default = 0, 0, 0
h_max_default, s_max_default, v_max_default = 179, 255, 255


# Create scales for HSV values in the HSV window
h_min_scale = Scale(hsv_window, from_=0, to=179, label="HMin", orient=tk.HORIZONTAL, length=200)
s_min_scale = Scale(hsv_window, from_=0, to=255, label="SMin", orient=tk.HORIZONTAL, length=200)
v_min_scale = Scale(hsv_window, from_=0, to=255, label="VMin", orient=tk.HORIZONTAL, length=200)
h_max_scale = Scale(hsv_window, from_=0, to=179, label="HMax", orient=tk.HORIZONTAL, length=200)
s_max_scale = Scale(hsv_window, from_=0, to=255, label="SMax", orient=tk.HORIZONTAL, length=200)
v_max_scale = Scale(hsv_window, from_=0, to=255, label="VMax", orient=tk.HORIZONTAL, length=200)

# Set default values for HSV scales
h_min_scale.set(h_min_default)
s_min_scale.set(s_min_default)
v_min_scale.set(v_min_default)
h_max_scale.set(h_max_default)
s_max_scale.set(s_max_default)
v_max_scale.set(v_max_default)

# Place HSV scales in the HSV window
h_min_scale.pack()
s_min_scale.pack()
v_min_scale.pack()
h_max_scale.pack()
s_max_scale.pack()
v_max_scale.pack()

# Bind HSV scales to update function
h_min_scale.bind("<ButtonRelease-1>", update_hsv)
s_min_scale.bind("<ButtonRelease-1>", update_hsv)
v_min_scale.bind("<ButtonRelease-1>", update_hsv)
h_max_scale.bind("<ButtonRelease-1>", update_hsv)
s_max_scale.bind("<ButtonRelease-1>", update_hsv)
v_max_scale.bind("<ButtonRelease-1>", update_hsv)

# Create a canvas in the HSV window to display HSV filtered result
hsv_canvas = tk.Canvas(hsv_window, width=img.shape[1], height=img.shape[0])
hsv_canvas.pack()

# Initialize HSV canvas with the original image
hsv_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
hsv_canvas.create_image(0, 0, anchor=tk.NW, image=hsv_photo)
hsv_canvas.img = hsv_photo

# Create a separate window for morphological operations
morph_window = tk.Toplevel()
morph_window.title("Morphological Operations")

# Create morphological operation sliders in the morph window
erode_iter_scale = Scale(morph_window, from_=0, to=10, label="Erode Iterations", orient=tk.HORIZONTAL, length=200)
dilate_iter_scale = Scale(morph_window, from_=0, to=10, label="Dilate Iterations", orient=tk.HORIZONTAL, length=200)
erode_kernel_scale = Scale(morph_window, from_=1, to=20, label="Erode Kernel Size", orient=tk.HORIZONTAL, length=200)
dilate_kernel_scale = Scale(morph_window, from_=1, to=20, label="Dilate Kernel Size", orient=tk.HORIZONTAL, length=200)

# Set default values for morphological operations
erode_iter_scale.set(1)
dilate_iter_scale.set(1)
erode_kernel_scale.set(3)
dilate_kernel_scale.set(3)

# Place morphological operation sliders in the morph window
erode_iter_scale.pack()
dilate_iter_scale.pack()
erode_kernel_scale.pack()
dilate_kernel_scale.pack()

# Bind morphological operation scales to update function
erode_iter_scale.bind("<ButtonRelease-1>", lambda event: update_hsv())
dilate_iter_scale.bind("<ButtonRelease-1>", lambda event: update_hsv())
erode_kernel_scale.bind("<ButtonRelease-1>", lambda event: update_hsv())
dilate_kernel_scale.bind("<ButtonRelease-1>", lambda event: update_hsv())

# Create a canvas in the morph window to display morphological operation result
morph_canvas = tk.Canvas(morph_window, width=img.shape[1], height=img.shape[0])
morph_canvas.pack()

# Initialize morph canvas with the original image
morph_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
morph_canvas.create_image(0, 0, anchor=tk.NW, image=morph_photo)
morph_canvas.img = morph_photo

hsv_window.mainloop()
