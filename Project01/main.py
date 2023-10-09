import PySimpleGUI as sg
import os.path
import io
import PIL.Image

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
    [sg.HorizontalSeparator(pad=(5, 10)), sg.Button("Save Image")]
]

filter_column = [
    [sg.Text("Filters:")],
    # [sg.Text(f'Displaying image: '), sg.Text(k='-FILENAME-')],
    # [sg.Image(key="-IMAGE-")],
    # [sg.Button("save")]
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
            window['-FILENAME-'].update(filename)
            image = PIL.Image.open(filename)
            resized_image = resize_image(image, max_width=500, max_height=500)
            image_data = convert_to_bytes(resized_image)
            window['-IMAGE-'].update(data=image_data)

        except Exception as e:
            print(e)

window.close()