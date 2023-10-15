import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from utils import *

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

    if len(img_array.shape) == 3:  # Imagem colorida (RGB)
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

def plot_color_histogram(image):
    pixels = np.array(image)

    # Plot histogram for each color channel
    fig, axs = plt.subplots(1, 3)

    for i, color in enumerate(['r', 'g', 'b']):
        axs[i].hist(pixels[:,:,i].ravel(), bins=256, color=color, alpha=0.7, histtype='step', linewidth=2)
        # axs[i].set_xlim(0, 256)
        # axs[i].set_ylim(0, pixels.shape[0]*pixels.shape[1]*0.5)
        axs[i].set_xlabel(f'Intensity - {color}')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'Histogram - {color.upper()}')

    plt.tight_layout()
    plt.show()

def equalize_histogram(image):
    if image.mode == 'L':
        grayscale_image = image.convert('L')
        equalized_image = equalize_histogram_grayscale(grayscale_image)
    elif image.mode == 'RGB':
        r, g, b = image.split()
        equalized_r = equalize_histogram_grayscale(r)
        equalized_g = equalize_histogram_grayscale(g)
        equalized_b = equalize_histogram_grayscale(b)
        equalized_image = PIL.Image.merge('RGB', (equalized_r, equalized_g, equalized_b))
    elif image.mode == 'RGBA':
        r, g, b, a = image.split()
        equalized_r = equalize_histogram_grayscale(r)
        equalized_g = equalize_histogram_grayscale(g)
        equalized_b = equalize_histogram_grayscale(b)
        equalized_image = PIL.Image.merge('RGBA', (equalized_r, equalized_g, equalized_b, a))
    else:
        raise ValueError("Unsupported image mode: {}".format(image.mode))

    return equalized_image

def equalize_histogram_grayscale(image):
    # Equalize histogram for a grayscale image using the provided method
    pixels = list(image.getdata())
    histogram, bins = np.histogram(pixels, bins=256, range=(0, 256), density=True)
    cdf = histogram.cumsum()
    cdf = 255 * cdf / cdf[-1]  # Normalize

    equalized_pixels = np.interp(pixels, bins[:-1], cdf)

    # Remodelar a imagem equalizada
    equalized_image = PIL.Image.new('L', image.size)
    equalized_image.putdata(equalized_pixels.astype(int))

    return equalized_image

def apply_custom_filter(image, custom_filter):
    # Converter a imagem para array numpy
    img_array = np.array(image)

    # Verificar se a imagem é colorida (tem 3 ou 4 canais de cor) ou escala de cinza
    if len(img_array.shape) == 3:
        # A imagem é colorida (RGB ou RGBA)
        filtered_image_array = apply_filter_color(img_array, custom_filter)
    else:
        # A imagem já está em escala de cinza
        filtered_image_array = apply_filter_grayscale(img_array, custom_filter)

    # Normalizar os valores para o intervalo [0, 255]
    filtered_image_array = (filtered_image_array / np.max(filtered_image_array)) * 255

    # Converter de volta para uint8
    filtered_image_array = filtered_image_array.astype('uint8')

    # Criar uma nova imagem a partir do array transformado
    if len(img_array.shape) == 3:
        # Se a imagem era colorida, manter os canais de cor
        filtered_image = PIL.Image.fromarray(filtered_image_array, 'RGB' if len(img_array.shape) == 3 else 'RGBA')
    else:
        # A imagem era em escala de cinza
        filtered_image = PIL.Image.fromarray(filtered_image_array, 'L')

    return filtered_image

def apply_filter_grayscale(image_array, custom_filter):
    # Aplicar o filtro por convolução em tons de cinza
    filtered_image_array = convolution(image_array, custom_filter)
    return filtered_image_array

def apply_filter_color(image_array, custom_filter):
    # Separar os canais de cor
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

    # Aplicar o filtro por convolução a cada canal de cor separadamente
    filtered_r = convolution(r, custom_filter)
    filtered_g = convolution(g, custom_filter)
    filtered_b = convolution(b, custom_filter)

    # Juntar os canais de volta
    filtered_image_array = np.stack((filtered_r, filtered_g, filtered_b), axis=-1)
    return filtered_image_array


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

def apply_high_boost(image, custom_filter, factor):
    # Converter a imagem para escala de cinza
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image)

    # Aplicar o filtro por convolução
    filtered_image_array = convolution(img_array, custom_filter)

    filtered_image_array = (factor - 1) * img_array + filtered_image_array

    filtered_image_array = normalize(filtered_image_array)

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

def laplacian_filter(image):
    kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])
    return apply_custom_filter(image, kernel)

def high_boost(image, factor):
    kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])
    return apply_high_boost(image, kernel, factor)

def apply_sobel(image, custom_filter):
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image)
    filtered_image_array = convolution(img_array, custom_filter)
    return filtered_image_array

def sobel_filter(image):
    kernel_x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    
    img_x_array = apply_sobel(image, kernel_x)
    img_y_array = apply_sobel(image, kernel_y)
    img_x_array = normalize(img_x_array)
    img_y_array = normalize(img_y_array)

    img_x = convert_to_image(img_x_array)
    img_y = convert_to_image(img_y_array)
    img_x = img_x.convert('L')
    img_y = img_y.convert('L')

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_x)
    plt.title('Derivada em X')

    plt.subplot(1, 2, 2)
    plt.imshow(img_y)
    plt.title('Derivada em y')

    plt.show()
    return image

def edge_detection(image):
    kernel_x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    
    img_x_array = apply_sobel(image, kernel_x)
    img_y_array = apply_sobel(image, kernel_y)

    img_array = np.sqrt(img_x_array ** 2 + img_y_array ** 2)
    img_array = normalize(img_array)
    return convert_to_image(img_array)

def fourier_transform(image):
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image)

    M, N = img_array.shape
    DFT = np.zeros((M, N), dtype=complex)

    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    DFT[u, v] += img_array[x, y] * np.exp(-2j * np.pi * (u * x / M + v * y / N))

    plt.imshow(np.abs(DFT), cmap='gray')
    plt.title('DFT da imagem (parte real)')
    plt.show()
    #return convert_to_image(normalize(DFT))
    return DFT

def inverse_fourier_transform(DFT):
    # grayscale_image = DFT.convert('L')
    # DFT = np.array(grayscale_image)

    M, N = DFT.shape
    image = np.zeros((M, N), dtype=float)

    for x in range(M):
        for y in range(N):
            for u in range(M):
                for v in range(N):
                    image[x, y] += DFT[u, v] * np.exp(2j * np.pi * (u * x / M + v * y / N)) / M * N

    return convert_to_image(normalize(image))

def apply_fft(image):
    #grayscale_image = image.convert('L')
    image = np.array(image)
    
    M, N = image.shape
    if M == 1 and N == 1:
        return image
    else:
        # Divide a imagem em quatro sub-imagens
        A = image[::2, ::2]
        B = image[::2, 1::2]
        C = image[1::2, ::2]
        D = image[1::2, 1::2]

        # Calcula as FFTs das sub-imagens
        A_hat = apply_fft(A)
        B_hat = apply_fft(B)
        C_hat = apply_fft(C)
        D_hat = apply_fft(D)

        # Combine as sub-imagens para obter a FFT da imagem completa
        top = np.hstack((A_hat + np.exp(-2j * np.pi * np.arange(M // 2) / M) * D_hat, A_hat + np.exp(-2j * np.pi * np.arange(M // 2) / M) * D_hat))
        bottom = np.hstack((B_hat + np.exp(-2j * np.pi * np.arange(M // 2) / M) * C_hat, B_hat - np.exp(-2j * np.pi * np.arange(M // 2) / M) * C_hat))
        return np.vstack((top, bottom))

def fft2_imp(image):
    fft_image = apply_fft(image)

    # Calcule o espectro de magnitude (parte real) da FFT
    magnitude_spectrum = np.abs(fft_image)

    # Exiba o espectro de magnitude
    plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
    plt.title('Espectro de Magnitude da FFT')
    plt.show()

    return convert_to_image(normalize(np.log1p(magnitude_spectrum)))

def fft2(image):
    image = np.array(image)
    fft_image = np.fft.fft2(image)

    # Calcule o espectro de magnitude (parte real) da FFT
    magnitude_spectrum = np.abs(fft_image)

    # Exiba o espectro de magnitude
    plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
    plt.title('Espectro de Magnitude da FFT')
    plt.show()
    return convert_to_image(normalize(np.log1p(magnitude_spectrum)))

def rgb_to_hsv_single_pixel(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    max_value = max(r, g, b)
    min_value = min(r, g, b)
    delta = max_value - min_value

    # Compute Hue
    if delta == 0:
        hue = 0
    elif max_value == r:
        hue = 60 * (((g - b) / delta) % 6)
    elif max_value == g:
        hue = 60 * (((b - r) / delta) + 2)
    else:
        hue = 60 * (((r - g) / delta) + 4)

    # Compute Saturation
    saturation = 0 if max_value == 0 else delta / max_value

    # Compute Value
    value = max_value

    return int(hue), int(saturation * 100), int(value * 100)

def rgb_to_hsv(image_rgb):
    width, height = image_rgb.size
    hsv_image = PIL.Image.new('HSV', (width, height))

    for x in range(width):
        for y in range(height):
            r, g, b, _ = image_rgb.getpixel((x, y))  # Ignora o canal alfa
            hsv_pixel = rgb_to_hsv_single_pixel(r, g, b)
            hsv_image.putpixel((x, y), hsv_pixel)

    return hsv_image

def hsv_to_rgb_single_pixel(h, s, v):
    h, s, v = h / 360.0, s / 100.0, v / 100.0

    i = int(h * 6)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if i % 6 == 0:
        r, g, b = v, t, p
    elif i % 6 == 1:
        r, g, b = q, v, p
    elif i % 6 == 2:
        r, g, b = p, v, t
    elif i % 6 == 3:
        r, g, b = p, q, v
    elif i % 6 == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 255), int(g * 255), int(b * 255)

def hsv_to_rgb(image_hsv):
    width, height = image_hsv.size
    rgb_image = PIL.Image.new('RGB', (width, height))

    for x in range(width):
        for y in range(height):
            h, s, v = image_hsv.getpixel((x, y))
            rgb_pixel = hsv_to_rgb_single_pixel(h, s, v)
            rgb_image.putpixel((x, y), rgb_pixel)

    return rgb_image

def apply_chroma_key_to_image(image_path, chroma_image_path):
    try:
        # Carregar a imagem original
        original_image = PIL.Image.open(image_path)

        # Definir o limiar de verde para Chroma Key (valores R, G, B)
        green_threshold = (0, 255, 0)

        # Aplicar o Chroma Key
        chroma_keyed_image = apply_chroma_key(original_image, green_threshold=green_threshold)

        # Carregar a imagem de chroma
        chroma_image = PIL.Image.open(chroma_image_path)

        # Substituir o fundo verde pelo conteúdo da imagem de chroma
        chroma_keyed_image.paste(chroma_image, (0, 0), chroma_image)

        # Exibir a imagem após o Chroma Key
        chroma_keyed_image.show()

    except Exception as e:
        print("Ocorreu um erro ao aplicar o Chroma Key:", str(e))

def apply_chroma_key(original_image, chroma_image, green_threshold=100):
    """
    Apply Chroma Key to the original image using the provided chroma image.
    
    Parameters:
        original_image (PIL.Image.Image): The original image.
        chroma_image (PIL.Image.Image): The chroma image to be used for Chroma Key.
        green_threshold (int): The green channel threshold for identifying the chroma color.
    
    Returns:
        PIL.Image.Image: The image after applying Chroma Key.
    """
    # Convert the chroma image to RGB
    chroma_image_rgb = chroma_image.convert("RGB")
    chroma_image_array = np.array(chroma_image_rgb)

    # Convert the original image to RGB
    original_image_array = np.array(original_image.convert("RGB"))
    
    # Resize the chroma image to match the dimensions of the original image
    chroma_image_array_resized = np.array(chroma_image.resize(original_image.size))

    # Define the mask for green color in the chroma image
    mask = (chroma_image_array_resized[:, :, 1] > green_threshold) & \
           (chroma_image_array_resized[:, :, 0] < green_threshold) & \
           (chroma_image_array_resized[:, :, 2] < green_threshold)

    # Replace the green color in the original image with the chroma image
    original_image_array[mask] = chroma_image_array_resized[mask]

    # Create a new image from the modified array
    chroma_keyed_image = PIL.Image.fromarray(original_image_array, 'RGB')

    return chroma_keyed_image