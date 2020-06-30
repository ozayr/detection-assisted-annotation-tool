import PySimpleGUI as sg
import os
import pickle
import sys

sg.theme('Dark Blue 15')

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def init_screen():
    try:
        file = open(os.path.join(__location__, 'init_configs'), 'rb')
        init_configs = pickle.load(file)
        file.close()
    except FileNotFoundError:
        init_configs = {}

    title = 'Detection Assisted Annotation Tool'
    layout = [

        [sg.Text(title, size=(len(title), 1), justification='center', font=("Helvetica", 15), relief=sg.RELIEF_RIDGE)],

        [sg.Text('choose image directory:'), sg.Input(init_configs.get('image_dir', ''), key='image_dir', size=(30, 1)),
         sg.FolderBrowse(target='image_dir', key='' + sg.WRITE_ONLY_KEY)],
        [sg.Checkbox('Edit Mode', default=init_configs.get('edit_mode', False), key='edit_mode'),
         sg.Checkbox('Augmentations', default=init_configs.get('do_augmentations', False), key='do_augmentations'),
         sg.Checkbox('save to COCO format', default=init_configs.get('convert_to_coco', False), key='convert_to_coco')],
        [sg.Checkbox('Test train split:', enable_events=True, key='test_train_split'),
         sg.Input('0.25', size=(5, 1), disabled=True, key='test_size'), sg.Text('image extension:'),
         sg.Input(init_configs.get('image_extension', '.jpg'), key='image_extension', size=(10, 1))],
        [sg.Button('Submit'), sg.Cancel()]
    ]

    init_screen = sg.Window('DAAT', layout, element_justification='center')
    while 1:
        event, values = init_screen.read()
        if event == 'Submit':
            if not os.path.exists(values['image_dir']):
                sg.popup('check image directory!')
            elif float(values['test_size']) < 0 or float(values['test_size']) > 1:
                sg.popup(f'check test size!\ncannot be {float(values["test_size"])} must be between 0 and 1')
            else:
                init_screen.close()
                file = open(os.path.join(__location__, 'init_configs'), 'wb')
                pickle.dump(values, file)
                file.close()
                return values
        elif event == 'test_train_split':
            if values['test_train_split']:
                init_screen['test_size'].update(disabled=False)
            else:
                init_screen['test_size'].update(disabled=True)
        elif event == 'Cancel':
            sys.exit()


def select_classes_screen(classes):
    layout = [
        [sg.Listbox(values=classes, key='classes', enable_events=True, size=(30, 5))],
        [sg.Text('Selected:'), sg.Text(key='selected_class', size=(20, 1))],
        [sg.Button('Submit')]
    ]
    class_screen = sg.Window('Select a class', layout)
    while 1:
        event, values = class_screen.read()
        if event == 'classes':
            class_screen['selected_class'].update(values['classes'][0])
        elif event == 'Submit':
            class_screen.close()
            return values['classes'][0]
        elif event == sg.WIN_CLOSED:
            class_screen.close()
            return 0


def add_new_class_screen():
    layout = [
        [sg.Text('Enter new class:'), sg.Input(key='new_class', size=(20, 1))],
        [sg.Submit(), sg.Cancel()],
    ]

    new_class_screen = sg.Window('Add new class', layout)
    while 1:
        event, values = new_class_screen.read()
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            return 0
        elif event == 'Submit':
            new_class_screen.close()
            return values['new_class']


def assign_hotkey_screen(classes):
    key_assignment = dict()
    layout = [
        [sg.Text('select a class then press a number (0-9) to assign:')],
        [sg.Listbox(values=classes, key='classes', enable_events=True, size=(30, 5))],
        [sg.Text('Selected:'), sg.Text(key='selected_class', size=(20, 1))],
        [sg.Text(key='assigned', size=(50, 3))],
        [sg.Submit()]

    ]

    hotkey_screen = sg.Window('Assign Hotkeys', layout, return_keyboard_events=True)

    while 1:
        event, values = hotkey_screen.read()
        if event == sg.WIN_CLOSED:
            hotkey_screen.close()
            return 0

        elif event == 'classes':
            hotkey_screen['selected_class'].update(values['classes'][0])

        elif event == 'Submit':
            hotkey_screen.close()
            key_assignment = {(i + 48): vals for i, vals in enumerate(key_assignment.values())}
            return key_assignment

        elif event.split(':')[0].isdigit() and values['classes']:
            key = int(event.split(':')[0])
            if key in range(0, 10):
                key_assignment[key] = values['classes'][0]
                hotkey_screen['assigned'].update({i: key_assignment[i] for i in sorted(key_assignment)})
