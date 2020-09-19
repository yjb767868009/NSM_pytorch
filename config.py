conf = {
    "save_path": "E:/NSM/trained",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        'dataset_path': "E:/NSM/NSM/data1",
    },
    "model": {
        'model_name': 'NSM',
        'batch_size': 64,
        'encoder_nums': 4,
        'encoder_dims': [[419, 512, 512],
                         [156, 128, 128],
                         [2034, 512, 512],
                         [2048, 512, 512]],
        'encoder_activations': [['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu']],
        'encoder_keep_prob': 0.7
    },
}
