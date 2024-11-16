import tkinter as tk
from tkinter import Scale
import numpy as np
from PIL import Image, ImageTk
import cv2

def update_hsv(event):
    h_min = h_min_scale.get()
    s_min = s_min_scale.get()
    v_min = v_min_scale.get()
    h_max = h_max_scale.get()
    s_max = s_max_scale.get()
    v_max = v_max_scale.get()

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(img, img, mask=mask)
 
    result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(result))
    canvas.img = photo
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

root = tk.Tk()
root.title("HSV Color Picker")

# Load image
img = cv2.imread("duckie.jpg")
img = cv2.resize(img, (480, 360))

# Create canvas to display the image
canvas = tk.Canvas(root, width=img.shape[1], height=img.shape[0])
canvas.pack()

# Default HSV values
h_min_default, s_min_default, v_min_default = 0, 0, 0
h_max_default, s_max_default, v_max_default = 179, 255, 255

# Create scales for HSV values
h_min_scale = Scale(root, from_=0, to=179, label="HMin", orient=tk.HORIZONTAL, length=200)
s_min_scale = Scale(root, from_=0, to=255, label="SMin", orient=tk.HORIZONTAL, length=200)
v_min_scale = Scale(root, from_=0, to=255, label="VMin", orient=tk.HORIZONTAL, length=200)
h_max_scale = Scale(root, from_=0, to=179, label="HMax", orient=tk.HORIZONTAL, length=200)
s_max_scale = Scale(root, from_=0, to=255, label="SMax", orient=tk.HORIZONTAL, length=200)
v_max_scale = Scale(root, from_=0, to=255, label="VMax", orient=tk.HORIZONTAL, length=200)

# Set default values
h_min_scale.set(h_min_default)
s_min_scale.set(s_min_default)
v_min_scale.set(v_min_default)
h_max_scale.set(h_max_default)
s_max_scale.set(s_max_default)
v_max_scale.set(v_max_default)

# Place scales on the window
h_min_scale.pack()
s_min_scale.pack()
v_min_scale.pack()
h_max_scale.pack()
s_max_scale.pack()
v_max_scale.pack()

# Bind scales to update function
h_min_scale.bind("<ButtonRelease-1>", update_hsv)
s_min_scale.bind("<ButtonRelease-1>", update_hsv)
v_min_scale.bind("<ButtonRelease-1>", update_hsv)
h_max_scale.bind("<ButtonRelease-1>", update_hsv)
s_max_scale.bind("<ButtonRelease-1>", update_hsv)
v_max_scale.bind("<ButtonRelease-1>", update_hsv)

# Initialize canvas with default image
photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
canvas.create_image(0, 0, anchor=tk.NW, image=photo)
canvas.img = photo

root.mainloop()
