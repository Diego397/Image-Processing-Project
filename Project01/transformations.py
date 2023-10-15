import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def apply_negative(image):
    # Converter a imagem para array numpy
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:

        # Separar os canais de cor RGB
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        # Inverter os valores de cada canal RGB
        negative_red_channel = 255 - red_channel
        negative_green_channel = 255 - green_channel
        negative_blue_channel = 255 - blue_channel

        # Criar uma nova imagem a partir dos arrays modificados
        negative_img_array = np.stack((negative_red_channel, negative_green_channel, negative_blue_channel), axis=-1)
    else:
        negative_img_array = 255 - img_array

    negative_image = PIL.Image.fromarray(negative_img_array.astype('uint8'))

    return negative_image

def apply_logarithmic_transformation(image, c):
    # Converter a imagem para array numpy
    img_array = np.array(image)

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Imagem colorida (RGB)
        # Separar os canais de cor RGB
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        # Aplicar a transformação logarítmica em cada canal
        log_transformed_red = c * np.log1p(red_channel)  # Usamos log1p para evitar log(0)
        log_transformed_green = c * np.log1p(green_channel)
        log_transformed_blue = c * np.log1p(blue_channel)

        # Normalizar os valores para o intervalo [0, 255] para cada canal
        log_transformed_red = (log_transformed_red / np.max(log_transformed_red)) * 255
        log_transformed_green = (log_transformed_green / np.max(log_transformed_green)) * 255
        log_transformed_blue = (log_transformed_blue / np.max(log_transformed_blue)) * 255

        # Empilhar os canais transformados de volta em uma imagem RGB
        transformed_img_array = np.stack((log_transformed_red, log_transformed_green, log_transformed_blue), axis=-1)
    else:  # Imagem em escala de cinza
        # Aplicar a transformação logarítmica
        log_transformed = c * np.log1p(img_array)  # Usamos log1p para evitar log(0)

        # Normalizar os valores para o intervalo [0, 255]
        log_transformed = (log_transformed / np.max(log_transformed)) * 255

        # Usar a transformação logarítmica em escala de cinza diretamente
        transformed_img_array = log_transformed

    # Lidar com valores infinitos ou NaN, substituindo-os por 0
    transformed_img_array = np.nan_to_num(transformed_img_array)

    # Garantir que os valores estejam no intervalo [0, 255]
    transformed_img_array = np.clip(transformed_img_array, 0, 255)

    # Criar uma nova imagem a partir do array transformado
    transformed_image = PIL.Image.fromarray(transformed_img_array.astype('uint8'))

    return transformed_image

def apply_gamma_correction(image, gamma):
    # Converter a imagem para array numpy
    img_array = np.array(image)

    if len(img_array.shape) == 3:  # Imagem colorida (RGB)
        # Separar os canais de cor RGB
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        # Aplicar a correção gama em cada canal
        gamma_corrected_red = np.power(red_channel / 255.0, gamma) * 255.0
        gamma_corrected_green = np.power(green_channel / 255.0, gamma) * 255.0
        gamma_corrected_blue = np.power(blue_channel / 255.0, gamma) * 255.0

        # Garantir que os valores estejam no intervalo [0, 255] para cada canal
        gamma_corrected_red = np.clip(gamma_corrected_red, 0, 255)
        gamma_corrected_green = np.clip(gamma_corrected_green, 0, 255)
        gamma_corrected_blue = np.clip(gamma_corrected_blue, 0, 255)

        # Empilhar os canais corrigidos de volta em uma imagem RGB
        gamma_corrected_img_array = np.stack((gamma_corrected_red, gamma_corrected_green, gamma_corrected_blue), axis=-1)
    else:  # Imagem em escala de cinza
        # Aplicar a correção gama
        gamma_corrected_img_array = np.power(img_array / 255.0, gamma) * 255.0

        # Garantir que os valores estejam no intervalo [0, 255]
        gamma_corrected_img_array = np.clip(gamma_corrected_img_array, 0, 255)

    # Criar uma nova imagem a partir do array corrigido
    gamma_corrected_image = PIL.Image.fromarray(gamma_corrected_img_array.astype('uint8'))

    return gamma_corrected_image

def apply_piecewise_linear_transformation(image, segments):
    # Converter a imagem para array numpy
    img_array = np.array(image)

    if len(img_array.shape) == 3:  # Imagem colorida (RGB)
        # Separar os canais de cor RGB
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        # Inicializar matrizes para armazenar os canais transformados
        transformed_red = np.zeros_like(red_channel)
        transformed_green = np.zeros_like(green_channel)
        transformed_blue = np.zeros_like(blue_channel)

        # Aplicar a transformação linear definida por partes em cada canal
        for start, end, slope in segments:
            mask = (red_channel >= start) & (red_channel <= end)
            transformed_red[mask] = slope * (red_channel[mask] - start)

            mask = (green_channel >= start) & (green_channel <= end)
            transformed_green[mask] = slope * (green_channel[mask] - start)

            mask = (blue_channel >= start) & (blue_channel <= end)
            transformed_blue[mask] = slope * (blue_channel[mask] - start)

        # Normalizar os valores para o intervalo [0, 255]
        transformed_red = np.clip(transformed_red, 0, 255)
        transformed_green = np.clip(transformed_green, 0, 255)
        transformed_blue = np.clip(transformed_blue, 0, 255)

        # Empilhar os canais transformados de volta em uma imagem RGB
        transformed_img_array = np.stack((transformed_red, transformed_green, transformed_blue), axis=-1)
    else:  # Imagem em escala de cinza
        # Inicializar uma matriz para armazenar a imagem transformada
        transformed_img_array = np.zeros_like(img_array)

        # Aplicar a transformação linear definida por partes
        for i, segment in enumerate(segments):
            start, end, slope = segment
            mask = (img_array >= start) & (img_array <= end)
            transformed_img_array[mask] = slope * (img_array[mask] - start)

        # Normalizar os valores para o intervalo [0, 255]
        transformed_img_array = np.clip(transformed_img_array, 0, 255)

    # Criar uma nova imagem a partir do array transformado
    transformed_image = PIL.Image.fromarray(transformed_img_array.astype('uint8'))

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

def apply_binarization(image, threshold):
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image)
    
    # Aplicar a binarização
    binarized_array = np.where(img_array >= threshold, 255, 0)
    
    # Criar uma nova imagem binarizada
    binarized_image = PIL.Image.fromarray(binarized_array.astype('uint8'))
    
    return binarized_image

def apply_mean_smoothing(image):
    img_array = np.array(image)

    # Aplicar a suavização da média usando um kernel 3x3
    kernel = np.ones((3, 3)) / 9
    smoothed_array = convolution(img_array, kernel)

    # Normalizar os valores para o intervalo [0, 255]
    smoothed_array = (smoothed_array / np.max(smoothed_array)) * 255

    # Converter de volta para uint8
    smoothed_array = smoothed_array.astype('uint8')

    # Criar uma nova imagem a partir do array suavizado
    smoothed_image = PIL.Image.fromarray(smoothed_array)

    return smoothed_image

def apply_weighted_mean_smoothing(image, custom_kernel):
    img_array = np.array(image)

    # Normalizar o kernel para garantir que a soma seja 1
    kernel_sum = np.sum(custom_kernel)
    normalized_kernel = custom_kernel / kernel_sum if kernel_sum != 0 else custom_kernel

    # Aplicar a suavização da média ponderada usando o kernel fornecido
    smoothed_array = convolution(img_array, normalized_kernel)

    # Normalizar os valores para o intervalo [0, 255]
    smoothed_array = (smoothed_array / np.max(smoothed_array)) * 255

    # Converter de volta para uint8
    smoothed_array = smoothed_array.astype('uint8')

    # Criar uma nova imagem a partir do array suavizado
    smoothed_image = PIL.Image.fromarray(smoothed_array)

    return smoothed_image

def apply_median_filter(image, filter_size):
    # Converter a imagem para escala de cinza
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image)

    # Aplicar a filtragem pela mediana
    filtered_image_array = median_filter(img_array, filter_size)

    # Converter de volta para uint8
    filtered_image_array = filtered_image_array.astype('uint8')

    # Criar uma nova imagem a partir do array transformado
    filtered_image = PIL.Image.fromarray(filtered_image_array)

    return filtered_image

def median_filter(image_array, filter_size):
    height, width = image_array.shape
    pad = filter_size // 2

    # Aplicar zero padding
    padded_image = np.pad(image_array, pad, mode='constant')

    # Inicializar a matriz resultante
    result = np.zeros((height, width))

    # Realizar a filtragem pela mediana
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+filter_size, j:j+filter_size]
            result[i, j] = np.median(region)

    return result