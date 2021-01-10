from recording import record, addingLine
import pathlib
import uuid

DATA_DIR = 'data/spanish_speech_commands'

if __name__ == '__main__':
    while True:
        filename_wav = str(uuid.uuid4())
        record(DATA_DIR + f'/silence/{filename_wav}', seconds=3, pause= False)
        addingLine(text=filename_wav, filename='none')