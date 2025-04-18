from PIL import Image
from collections import Counter
import webcolors
import cv2 
from colorthief import ColorThief
from io import BytesIO
import numpy as np
from sklearn.cluster import KMeans
########################     ########################################
# the correct and "smart" method to extract the colors with use of clustring and dominant colors
########################     ########################################

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def dominant_color(image):
     # Convert NumPy array to PIL Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    # Open image and resize for efficiency
    image = image.resize((50, 50))
    
    # Get colors from image
    pixels = image.getcolors(50 * 50)
    
    # Sort them by count/frequency and get the most frequent color
    most_frequent_pixel = pixels[0]
    for count, color in pixels:
        if count > most_frequent_pixel[0]:
            most_frequent_pixel = (count, color)
    
    # Map color to its name
    color_name = closest_color(most_frequent_pixel[1])
    
    # Return the color name
    return most_frequent_pixel[1]

def dominant_color_cluster(image, n_clusters=3):
    # Convert NumPy array to PIL Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    # Resize the image for efficiency
    image = image.resize((50, 50))
    
    # Convert the image to a 2D array of pixels
    pixels = np.array(image).reshape(-1, 3)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)
    
    # Compute the average color of each cluster
    avg_colors = kmeans.cluster_centers_
    
    # Find the cluster with the highest count
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]
    
    # Get the dominant color
    dominant_color = avg_colors[dominant_cluster]
    
    return dominant_color

def three_most_dominant_colors(image):
    # Convert NumPy array to PIL Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    # Save the image in memory
    image_file = BytesIO()
    image.save(image_file, format='JPEG')
    image_file.seek(0)
    
    # Get the dominant colors using ColorThief
    color_thief = ColorThief(image_file)
    dominant_colors = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=3, quality=1)
    
    return dominant_colors, palette

def average_color(image, bounding_box):
    x1, y1, x2, y2 = bounding_box
    cropped_image = image[y1:y2, x1:x2]
    avg_color = np.mean(cropped_image, axis=(0, 1))
    return avg_color