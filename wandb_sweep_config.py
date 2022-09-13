import math

sweep_config = {
    # 'method': 'random',
    'method': 'grid',

    'metric': {'name': 'avg_loss_test', 'goal': 'minimize'},

    'parameters': {
                    # 'model': {'values': ['ViT', 'resnet']},
                    'model': {'value': 'ViT'},
                    # 'model': {'value': 'resnet'},

                    # 'early_stopping': {'values': [True, False]},
                    'early_stopping': {'value': True},
                    # 'early_stopping': {'value': False},

                    'epochs': {'value': 60},
                    # 'epochs': {'value': 2},

                    # 'optimizer': {'values': ['adam', 'sgd']},
                    'optimizer': {'value': 'adam'},

                    # 'fc_layer_size': {'values': [128, 256, 512]},
                    # 'dropout': {'values': [0, 0.1, 0.3]},
                    'dropout': {'value': 0},
                    'l2_regularization_weight': {'values': [0, 0.03, 0.08, 0.15]},

                    # 'learning_rate': {'values': [0.001, 0.01, 0.03, 0.1]},
                    'learning_rate': {'value': 0.01},

                    # 'learning_rate': {
                    #         # a flat distribution between 0 and 0.1
                    #         'distribution': 'uniform',
                    #         'min': 0.0001,
                    #         'max': 0.002
                    #       },

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
def get_example_set_hyperparams(wand_sweep_config):
    example_hyperparams = {}
    wand_hyperparameters = wand_sweep_config['parameters']
    for k in wand_hyperparameters:
        if 'value' in wand_hyperparameters[k]:
            example_hyperparams[k] = wand_hyperparameters[k]['value']
        elif 'values' in wand_hyperparameters[k]:
            example_hyperparams[k] = wand_hyperparameters[k]['values'][0]
    return example_hyperparams


if __name__ == "__main__":
    hyperparams = get_example_set_hyperparams()
    print('hyperparams', hyperparams)