import numpy as np
import PIL.Image

def normalize(img_array):
    if np.min(img_array) < 0:
        img_array += abs(np.min(img_array))
    img_array = (img_array / np.max(img_array)) * 255
    return img_array

def convert_to_image(img_array):
    img_array = img_array.astype('uint8')
    image = PIL.Image.fromarray(img_array)
    return image