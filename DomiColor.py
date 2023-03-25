from PIL import Image
from collections import Counter
import webcolors
import cv2 
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


