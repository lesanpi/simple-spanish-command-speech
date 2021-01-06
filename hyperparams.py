class Hyperparams:

    prepro = False

    vocab = "PE abcdefghigklmnñopqrstuvwxyz{.¿?"
    data = "data/tacotron/"

    test_data = ""
    max_duration = 10.0

    # Procesamiento de señales
    sr = 22050 # Sample rate.
    n_fft = 2048 # fft muestras
    frame_shift = 0.0125 # segundos
    frame_length = 0.05 # segundos
    hop_length = int(sr*frame_shift)
    win_length = int(sr*frame_length)
    n_mels = 80
    power = 1.2
    n_iter = 50
    preemphasis = .97
    max_db = 100
    ref_db = 20

    # Modelo
    lr = 0.001
    logdir = ""
    sampledir = "samples"
    batch_size = 32
    r = 5 # Factor de reduction
    dropout_rate = .5

    # training scheme
    lr = 0.001  # Initial learning rate.
    logdir = "logdir/01"
    sampledir = 'samples'
    batch_size = 32
    num_epochs = 30