conf = {
    "save_path": "./work",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        'dataset_path': "your_dataset_path",
        'resolution': '64',
        'input_dim': 5307,
        'load_path': '../../Export',
        'save_path': '../../trained',
    },
    "model": {
        'model_name': 'NSM',
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
