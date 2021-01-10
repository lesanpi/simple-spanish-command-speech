import uuid
from main import *
from tensorflow.keras.models import load_model
import pyaudio
import wave
import os
import tensorflow as tf
import numpy as np
import pathlib
from test import *

model = load_model('models/model.h5')
sample_file = pathlib.Path('test/test.wav')

main_commands = ['abajo','alto','arriba','derecha', 'izquierda','no','si','sigue']
DATA_DIR = pathlib.Path('data/spanish_speech_commands')
COMMANDS = np.array(tf.io.gfile.listdir(str(DATA_DIR)))
COMMANDS = COMMANDS[COMMANDS != 'README.md']

WAVS_PATH = "./data/spanish_speech_commands/"
PERSONS_PATH = "./data/persons/"
chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 4000  # Record at 44100 samples per second

def commandsToRecord():
    files = tf.io.gfile.glob(str(DATA_DIR) + '/*/*')
    commandsToRecord = []
    for command in COMMANDS:
        commandFiles = tf.io.gfile.glob(str(DATA_DIR) + f'/{command}/*')

        if len(commandFiles) / len(files) < 1 / len(COMMANDS) and command != 'none':
            print(command, len(commandFiles) / len(files), 1 / len(COMMANDS))
            commandsToRecord.append(command)
    if not commandsToRecord and len(files) < len(COMMANDS) * 100:
        commandsToRecord = COMMANDS

    return commandsToRecord

def preprocess(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram,  num_parallel_calls=AUTOTUNE)
  return output_ds

def record(filename: str, seconds: int):
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    input("Presiona enter para empezar a grabar..")
    print('Grabando...')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for seconds + 1
    for i in range(0, int(fs / chunk * (seconds + 1))):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Grabacion terminada')

    # Save the recorded data as a WAV file
    wf = wave.open(filename + ".wav", 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))


    wf = wave.open("test/test.wav", 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("Audio guardado.")


def addingLine(text: str, filename: str):
    text = text + " \n"
    try:
        file = open(PERSONS_PATH + filename + ".txt", "r")
    except:
        file = open(PERSONS_PATH + filename + ".txt", "w")
        file.close()
        file = open(PERSONS_PATH + filename + ".txt", "r")
    lines = file.readlines()
    file.close()

    file = open(PERSONS_PATH + filename + ".txt", "w")
    lines.append(text)
    file.writelines(lines)
    file.close()


def recordData(dir, command):
    dialog = command
    seconds = 3
    filename_txt = dir

    try:
        file = open(PERSONS_PATH + filename_txt + ".txt", "r")
    except:

        file = open(PERSONS_PATH + filename_txt + ".txt", "w")
        file.write(" ")
        file.close()
        file = open(PERSONS_PATH + filename_txt + ".txt", "r")

    lines = file.readlines()
    numLines = len(lines)

    filename_wav = WAVS_PATH + command + '/' + str(uuid.uuid4())

    record(filename=filename_wav, seconds=seconds)
    addingLine(text=filename_wav, filename=filename_txt)
    sample_ds = preprocess([str(sample_file)])

    for spectrogram in sample_ds.batch(1):
        prediction = model(spectrogram)
        pre = np.argmax(model.predict(spectrogram), axis=1)
        print("Prediction:", commands[pre])
        #plt.bar(commands, tf.nn.softmax(prediction[0]))
        #plt.title(f'Predictions for {command}')
        #plt.show()


if __name__ == '__main__':
    start = "y"
    dir = input("Quien esta hablando? ").lower()
    try:
        for com in main_commands:
            os.mkdir(f'./data/spanish_speech_commands/{com}')
    except:
        pass

    while start == "y":

        for command in commandsToRecord():
            for i in range(5):
                print()
                print(f"Grabando \"{command}\"...")
                recordData(dir=dir, command=command)

        start = input("Continuar? (y) n: ")
        if start == "y" or start == "":
            start = "y"