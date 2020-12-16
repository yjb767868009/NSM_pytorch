conf = {
    "CUDA_VISIBLE_DEVICES": "0",
    "data_root": "E:/NSM/data2",
    "base_model": {
        'model_name': 'nsm',
        'epoch': 10,
        'batch_size': 4,
        'lr': 0.001,
        'segmentation': [0, 419, 575, 2609, 4657, 5307],

        "save_path": "E:/NSM/trained",
        "load_path": "E:/NSM/trained",
        "save_output": "E:/NSM/data2",

        'encoder_nums': 5,
        'encoder_dims': [[419, 512, 512],
                         [156, 128, 128],
                         [2034, 512, 512],
                         [2048, 512, 512],
                         [650, 512, 512]],
        'encoder_activations': [['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu']],
        'encoder_dropout': 0.3,

        'rnn_dims': [2176, 2048, 2048, 2048, 2048, 2048, 1024, 618],
        'rnn_activations': ['elu', 'elu', 'elu'],
        'rnn_dropout': 0.3,
    },
    "gan_model": {
        'model_name': 'GAN',
        'epoch': 100,
        'batch_size': 8,
        'lr': 0.001,

        "save_path": "E:/NSM/trained",
        "load_path": "E:/NSM/trained",

        'refiner_dims': [618, 512, 512, 618],
        'refiner_activations': ['elu', 'elu'],
        'refiner_dropout': 0.3,

        'discriminative_dims': [618, 512, 512, 2, 1],
        'discriminative_activations': ['elu', 'elu', 'Sigmoid'],
        'discriminative_dropout': 0.3,
    },
}
