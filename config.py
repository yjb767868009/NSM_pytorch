conf = {
    "save_path": "E:/NSM/trained",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        'dataset_path': "E:/NSM/NSM/data1",
    },
    "model": {
        'model_name': 'NSM',
        'epoch': 150,
        'batch_size': 64,
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
        'encoder_keep_prob': 0.7,
        'expert_components': [1, 10],
        'expert_dims': [[650, 512, 512, 10],
                        [1664, 512, 512, 618]],
        'expert_activations': [['elu', 'elu', 'softmax'],
                               ['elu', 'elu', None]],
        'expert_keep_prob': 0.7,
        'lr': 0.0001,

    },
}
