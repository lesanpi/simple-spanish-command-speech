import os
import pathlib
data_dir = pathlib.Path('data/spanish_speech_commands')

def add_new_command(command):
    if not os.path.exists(data_dir/f'{command}'):
        os.mkdir(data_dir/f'{command}')

if __name__ == '__main__':
    command = input("Ingresa el nuevo comando: ")
    add_new_command(command)