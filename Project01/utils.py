import numpy as np
import PIL.Image
import io

def normalize(img_array):
    if np.min(img_array) < 0:
        img_array += abs(np.min(img_array))
    img_array = (img_array / np.max(img_array)) * 255
    return img_array

def convert_to_image(img_array):
    img_array = img_array.astype('uint8')
    image = PIL.Image.fromarray(img_array)
    return image


def convert_to_bytes(image, format="PNG"):
    with io.BytesIO() as bio:
        image.save(bio, format=format)
        return bio.getvalue()

def resize_image(image, max_width, max_height):
    width, height = image.size
    if width > max_width or height > max_height:
        aspect_ratio = width / height
        if width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        image = image.resize((new_width, new_height))
    return image