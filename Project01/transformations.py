import numpy as np
import PIL.Image

def apply_negative(image):
    # Converter a imagem para array numpy
    img_array = np.array(image)

    # print(img_array)
        
    # Inverter os valores dos canais RGB
    inverted_img_array = 255 - img_array

    # print(inverted_img_array)
    
    # Criar uma nova imagem a partir do array invertido
    negative_image = PIL.Image.fromarray(inverted_img_array.astype('uint8'))

    return negative_image

def apply_logarithmic_transformation(image, c):
    # Converter a imagem para array numpy
    img_array = np.array(image)

    # Aplicar a transformação logarítmica
    log_transformed = c * np.log(1 + img_array)

    # Normalizar os valores para o intervalo [0, 255]
    log_transformed = (log_transformed / np.max(log_transformed)) * 255

    # Converter de volta para uint8
    log_transformed = log_transformed.astype('uint8')

    # Criar uma nova imagem a partir do array transformado
    transformed_image = PIL.Image.fromarray(log_transformed)

    return transformed_image

def apply_gamma_correction(image, gamma):
    # Converter a imagem para array numpy
    img_array = np.array(image)

    # Aplicar a correção gama
    gamma_corrected = np.power(img_array / 255.0, gamma) * 255.0

    # Converter de volta para uint8
    gamma_corrected = gamma_corrected.astype('uint8')

    # Criar uma nova imagem a partir do array transformado
    corrected_image = PIL.Image.fromarray(gamma_corrected)

    return corrected_image

def apply_piecewise_linear_transformation(image, segments):
    # Converter a imagem para array numpy
    img_array = np.array(image)

    # Aplicar a transformação linear definida por partes
    transformed_img_array = np.piecewise(img_array, [
        (img_array >= segments[0][0]) & (img_array < segments[0][1]),
        (img_array >= segments[1][0]) & (img_array < segments[1][1]),
        # Adicione mais condições para outros segmentos, se necessário
    ], [
        lambda x: segments[0][2] * (x - segments[0][0]),
        lambda x: segments[1][2] * (x - segments[1][0]),
        # Adicione mais transformações para outros segmentos, se necessário
    ], 0)

    # Normalizar os valores para o intervalo [0, 255]
    transformed_img_array = np.clip(transformed_img_array, 0, 255)

    # Converter de volta para uint8
    transformed_img_array = transformed_img_array.astype('uint8')

    # Criar uma nova imagem a partir do array transformado
    transformed_image = PIL.Image.fromarray(transformed_img_array)

    return transformed_image