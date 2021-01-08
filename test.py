import pathlib
from recording import record
from main import *
from tensorflow.keras.models import load_model

model = load_model('models/model.h5')
sample_file = pathlib.Path('test/test.wav')

def preprocess(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram,  num_parallel_calls=AUTOTUNE)
  return output_ds

if __name__ == '__main__':
  while True:
    record('test/test', seconds=2)
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram in sample_ds.batch(1):
      prediction = model(spectrogram)
      plt.bar(commands, tf.nn.softmax(prediction[0]))
      plt.title(f'Predictions')
      plt.show()