conf = {
    "save_path": "E:/AI4Animation-master/AI4Animation/SIGGRAPH_Asia_2019/trained",
    "load_path": "E:/AI4Animation-master/AI4Animation/SIGGRAPH_Asia_2019/trained",
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'dataset_path': "E:/AI4Animation-master/AI4Animation/SIGGRAPH_Asia_2019/Export",
    },
    "model": {
        'model_name': 'NSM',
        'epoch': 11,
        'batch_size': 1,
        'segmentation': [0, 419, 575, 2609, 4657, 5307],
        'encoder_nums': 4,
        'encoder_dims': [[419, 512, 512],
                         [156, 128, 128],
                         [2034, 512, 512],
                         [2048, 512, 512]],
        'encoder_activations': [['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu']],
        'encoder_dropout': 0.3,

        'rnn_dims': [1664, 512, 512, 512, 512, 618],
        'rnn_activations': ['elu', 'elu', 'elu'],
        'rnn_dropout': 0.3,

        'refiner_dims': [618, 512, 512, 618],
        'refiner_activations': ['elu', 'elu'],
        'refiner_dropout': 0.3,

        'discriminative_dims': [618, 512, 512, 2, 1],
        'discriminative_activations': ['elu', 'elu', 'elu'],
        'discriminative_dropout': 0.3,

        'lr': 0.001,
    },
}
