import math

sweep_config = {
    'method': 'random',

    'metric': {'name': 'loss', 'goal': 'minimize'},

    'parameters': {
                    # 'model': {'values': ['ViT', 'resnet']},
                    'model': {'value': 'ViT'},
                    # 'model': {'value': 'resnet'},

                    # 'early_stopping': {'values': [True, False]},
                    'early_stopping': {'value': True},
                    # 'early_stopping': {'value': False},

                    'epochs': {'value': 60},
                    # 'epochs': {'value': 2},

                    'optimizer': {'values': ['adam', 'sgd']},
                    # 'fc_layer_size': {'values': [128, 256, 512]},
                    # 'dropout': {'values': [0, 0.1, 0.3]},
                    'dropout': {'value': 0},
                    'l2_regularization_weight': {'values': [0, 0.3, 0.08, 0.15]},

                    'learning_rate': {
                            # a flat distribution between 0 and 0.1
                            'distribution': 'uniform',
                            'min': 0.0001,
                            'max': 0.002
                          },
                        'batch_size': {
                            'value': 10
                            # integers between 32 and 256
                            # with evenly-distributed logarithms
                            # 'distribution': 'grid',
                            # 'q': 1,
                            # 'min': math.log(32),
                            # 'max': math.log(256),
                          }
                    }
    }