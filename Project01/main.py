import PySimpleGUI as sg
import os.path
import io
import PIL.Image
from transformations import *

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


# Primeira coluna
file_list_column = [   
    [sg.Text("Image Folder"), sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"), sg.FolderBrowse()],
    [sg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-")]
]

# A coluna para exibir a imagem
image_viewer_column = [
    [sg.Text("Choose an image from list on the left:")],
    [sg.Text(f'Displaying image: '), sg.Text(k='-FILENAME-')],
    [sg.Image(key="-IMAGE-")],
    [sg.HorizontalSeparator(pad=(5, 10)), sg.Button("Save Image")],
]

filter_column = [
    [sg.Text("Filters:")],
    [sg.Button("Negative")],
    [sg.Button("Logarithmic Transformation")],
    [sg.Text("Constant 'c':"), sg.InputText(key="-C-")],
    [sg.Button("Gamma Correction")],
    [sg.Text("Gamma Value:"), sg.InputText(key="-GAMMA-")],
    [sg.Button("Piecewise Linear Transformation")],
    [sg.Text("Segment 1 Start:"), sg.InputText(key="-SEGMENT1_START-")],
    [sg.Text("Segment 1 End:"), sg.InputText(key="-SEGMENT1_END-")],
    [sg.Text("Segment 1 Slope:"), sg.InputText(key="-SEGMENT1_SLOPE-")],
]

# ----- Layout completo -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
        sg.VSeperator(),
        sg.Column(filter_column),
    ]
]

window = sg.Window("Image Viewer", layout, size=(1200, 675))


# Loop de eventos
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", ".bmp", ".jpg", ".tif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            lastfilename = os.path.basename(values["-FILE LIST-"][0])
            window['-FILENAME-'].update(lastfilename)
            image = PIL.Image.open(filename)
            resized_image = resize_image(image, max_width=500, max_height=500)
            image_data = convert_to_bytes(resized_image)
            window['-IMAGE-'].update(data=image_data)

        except Exception as e:
            print(e)

    elif event == "Negative":
        try:
            if "image" in locals():
                # Aplicar o negativo à imagem
                image = apply_negative(image)
                resized_image = resize_image(image, max_width=500, max_height=500)
                image_data = convert_to_bytes(resized_image)
                window['-IMAGE-'].update(data=image_data)

        except Exception as e:
            print(e)

    elif event == "Logarithmic Transformation":
        try:
            if "image" in locals():
                # Obter o valor de 'c' inserido pelo usuário
                c_value = float(values["-C-"]) if values["-C-"] else 1.0
        
                # Aplicar a transformação logarítmica à imagem com o valor de 'c'
                image = apply_logarithmic_transformation(image, c_value)
                resized_image = resize_image(image, max_width=500, max_height=500)
                image_data = convert_to_bytes(resized_image)
                window['-IMAGE-'].update(data=image_data)

        except ValueError:
            print("Por favor, insira um valor numérico válido para a constante 'c'.")

        except Exception as e:
            print(e)

    elif event == "Gamma Correction":
        try:
            if "image" in locals():
                # Obter o valor do gama inserido pelo usuário
                gamma_value = float(values["-GAMMA-"]) if values["-GAMMA-"] else 1.0
        
                # Aplicar a correção gama à imagem com o valor do gama
                image = apply_gamma_correction(image, gamma_value)
                resized_image = resize_image(image, max_width=500, max_height=500)
                image_data = convert_to_bytes(resized_image)
                window['-IMAGE-'].update(data=image_data)

        except ValueError:
            print("Por favor, insira um valor numérico válido para o gama.")

        except Exception as e:
            print(e)

    elif event == "Piecewise Linear Transformation":
        try:
            if "image" in locals():
                # Obter os valores inseridos pelo usuário para os segmentos
                segment1_start = float(values["-SEGMENT1_START-"])
                segment1_end = float(values["-SEGMENT1_END-"])
                segment1_slope = float(values["-SEGMENT1_SLOPE-"])

                # Definir os segmentos
                segments = [(segment1_start, segment1_end, segment1_slope)]
                # Adicione mais segmentos, se necessário
                
                # Aplicar a transformação linear definida por partes à imagem
                image = apply_piecewise_linear_transformation(image, segments)
                resized_image = resize_image(image, max_width=500, max_height=500)
                image_data = convert_to_bytes(resized_image)
                window['-IMAGE-'].update(data=image_data)

        except ValueError:
            print("Por favor, insira valores numéricos válidos para os segmentos.")

        except Exception as e:
            print(e)

window.close()