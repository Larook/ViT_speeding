import math

sweep_config = {
    'method': 'random',

    'metric': {'name': 'loss', 'goal': 'minimize'},

    'parameters': {

                    'epochs': {'value': 20},

                    'optimizer': {'values': ['adam', 'sgd']},
                    # 'fc_layer_size': {'values': [128, 256, 512]},
                    # 'dropout': {'values': [0.3, 0.4, 0.5]},

                    'learning_rate': {
                            # a flat distribution between 0 and 0.1
                            'distribution': 'uniform',
                            'min': 0,
                            'max': 0.1
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