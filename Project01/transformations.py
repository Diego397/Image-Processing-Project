import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

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

    # Inicializar uma matriz para armazenar a imagem transformada
    transformed_img_array = np.zeros_like(img_array)

    # Aplicar a transformação linear definida por partes
    for i, segment in enumerate(segments):
        start, end, slope = segment
        mask = (img_array >= start) & (img_array <= end)
        transformed_img_array[mask] = slope * (img_array[mask] - start)

    # Normalizar os valores para o intervalo [0, 255]
    transformed_img_array = np.clip(transformed_img_array, 0, 255)

    # Converter de volta para uint8
    transformed_img_array = transformed_img_array.astype('uint8')

    # Criar uma nova imagem a partir do array transformado
    transformed_image = PIL.Image.fromarray(transformed_img_array)

    return transformed_image

def hide_message(image, message):
    img_array = np.array(image)

    # Verificar se a imagem é colorida ou em escala de cinza
    is_color_image = len(img_array.shape) == 3 and img_array.shape[2] == 3

    # Transformar a mensagem em uma sequência de bits
    bits = ''.join(format(ord(c), '08b') for c in message)

    # Garantir que a mensagem cabe na imagem
    num_pixels_needed = len(bits)
    num_pixels_available = img_array.size // (3 if is_color_image else 1)  # Corrigindo o cálculo do número de pixels disponíveis

    if num_pixels_needed > num_pixels_available:
        raise ValueError("A mensagem é muito longa para a imagem fornecida.")
    
    while num_pixels_needed < num_pixels_available:
        bits += "0"
        num_pixels_needed += 1

    # Iterar sobre cada pixel e substituir os bits menos significativos pela mensagem
    bit_index = 0
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if is_color_image:
                for k in range(img_array.shape[2]):
                    if bit_index < len(bits):
                        img_array[i][j][k] = (img_array[i][j][k] & 0b11111110) | int(bits[bit_index])
                        bit_index += 1
                    else:
                        break
                else:
                    continue
                break
            else:
                if bit_index < len(bits):
                    img_array[i][j] = (img_array[i][j] & 0b11111110) | int(bits[bit_index])
                    bit_index += 1
                else:
                    break
        else:
            continue
        break

    # Criar uma nova imagem a partir do array modificado
    stego_image = PIL.Image.fromarray(img_array)

    return stego_image

def reveal_message(stego_image):
    img_array = np.array(stego_image)

    # Verificar se a imagem é colorida ou em escala de cinza
    is_color_image = len(img_array.shape) == 3 and img_array.shape[2] == 3

    # Inicializar a lista para armazenar os bits da mensagem
    bits = []

    # Extrair os bits menos significativos dos pixels
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if is_color_image:
                for k in range(img_array.shape[2]):
                    bits.append(img_array[i][j][k] & 1)
            else:
                bits.append(img_array[i][j] & 1)

    # Reconstruir a mensagem a partir dos bits
    message = ""
    byte = ""
    for bit in bits:
        byte += str(bit)
        if len(byte) == 8:
            message += chr(int(byte, 2))
            byte = ""

    return message

def plot_histogram(image):
    grayscale_image = image.convert('L')
    pixels = list(grayscale_image.getdata())
    plt.figure()
    plt.hist(pixels, bins=256)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def equalize_histogram(image):
    grayscale_image = image.convert('L')
    pixels = list(grayscale_image.getdata())

    # Calcular o histograma
    histogram = [0] * 256
    for pixel_value in pixels:
        histogram[pixel_value] += 1

    # Calcular a função de distribuição acumulada (CDF)
    cdf = [sum(histogram[:i+1]) for i in range(256)]
    cdf_normalized = [((cdf_val - min(cdf)) / (grayscale_image.width * grayscale_image.height - min(cdf))) * 255 for cdf_val in cdf]

    # Equalizar a imagem usando a CDF
    equalized_pixels = [int(cdf_normalized[pixel_value]) for pixel_value in pixels]

    # Remodelar a imagem equalizada
    equalized_image = PIL.Image.new('L', grayscale_image.size)
    equalized_image.putdata(equalized_pixels)

    return equalized_image

def apply_custom_filter(image, custom_filter):
    # Converter a imagem para escala de cinza
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image)

    # Aplicar o filtro por convolução
    filtered_image_array = convolution(img_array, custom_filter)

    if np.min(filtered_image_array) < 0:
        filtered_image_array += abs(np.min(filtered_image_array))

    # Normalizar os valores para o intervalo [0, 255]
    filtered_image_array = (filtered_image_array / np.max(filtered_image_array)) * 255

    # Converter de volta para uint8
    filtered_image_array = filtered_image_array.astype('uint8')

    # Criar uma nova imagem a partir do array transformado
    filtered_image = PIL.Image.fromarray(filtered_image_array)

    return filtered_image

def convolution(image_array, kernel):
    height, width = image_array.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Aplicar zero padding
    padded_image = np.pad(image_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Inicializar a matriz resultante
    result = np.zeros((height, width))

    # Realizar a convolução
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+k_height, j:j+k_width]
            result[i, j] = np.sum(region * kernel)

    return result