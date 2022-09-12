# CODE FROM https://towardsdatascience.com/a-demonstration-of-using-vision-transformers-in-pytorch-mnist-handwritten-digit-recognition-407eafbc15b0

import os
import pprint
import time

import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from einops import rearrange
from torch import nn, optim
import einops
from tqdm import tqdm

from SimulationDataLoader import SimulationDataLoader
from SimulationImageDataset import SimulationImageDataset
from model_training import vit_pytorch


class ViTRegression(nn.Module):
    def __init__(self, wandb_config, *, image_size, patch_size, num_outputs, dim, depth, heads, mlp_dim, channels=3):
        torch.manual_seed(42)

        super().__init__()

        self.wandb_config = wandb_config
        print('self.wandb_config', self.wandb_config)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        # self.device = torch.device(wandb_config['device'])

        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)).to(self.device)
        self.patch_to_embedding = nn.Linear(patch_dim, dim).to(self.device)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)).to(self.device)
        self.transformer = vit_pytorch.Transformer(dim, depth, heads, mlp_dim).to(self.device)

        self.to_cls_token = nn.Identity().to(self.device)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_outputs)
        ).to(self.device)
        # self.optimizer = optim.Adam(self.parameters(), lr=self.wandb_config['learning_rate'])  #0.003
        print("self.wandb_config")
        pprint.pprint(self.wandb_config)
        self.optimizer = self.build_optimizer(self.wandb_config['optimizer'])
        self.criterion = F.smooth_l1_loss

        self.no_outputs = self.mlp_head._modules['2'].out_features

        """ early stopping """
        self.trigger_times = 0
        self.patience = 3

    def build_optimizer(self, optimizer):
        if optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.wandb_config['learning_rate'], momentum=0.9, weight_decay=self.wandb_config['l2_regularization_weight'])
        elif optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.wandb_config['learning_rate'], weight_decay=self.wandb_config['l2_regularization_weight'])
        return optimizer

    def load_dataloaders(self, pickle_df_path):
        """ reads the pickles from the directory and updates dfs, SimulationData and dataloaders """
        # load training data
        self.data = SimulationDataLoader(create=False)
        self.train_df, self.test_df = self.data.load_dfs_from_pickles(create=False, training_percentage=0.8,
                                                                      pickle_df_path=pickle_df_path, shuffle=True)

        train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                               # transforms.Grayscale(num_output_channels=1),
                                               transforms.ColorJitter(brightness=.5, hue=.3),
                                               transforms.RandomEqualize(),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()
                                               ])

        dataset_train = SimulationImageDataset(main_df=self.train_df, transform=train_transforms)
        dataset_test = SimulationImageDataset(main_df=self.test_df, transform=train_transforms)

        # create dataloaders
        self.train_dataloader = DataLoader(dataset_train, batch_size=self.wandb_config['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(dataset_test, batch_size=self.wandb_config['batch_size'], shuffle=True)
        pass

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p).to(self.device)  # height weight channel batch
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        # print('cls_tokens', cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        # print('x', x)
        return self.mlp_head(x)

    def calculate_loss(self, output, target_angles, target_vels):
        if self.no_outputs == 2:
            target = torch.cat((target_angles.unsqueeze(1).float(), target_vels.unsqueeze(1).float()), 1)
            loss = self.criterion(output, target)  # L1 loss for regression applications
        else:
            target = target_angles.unsqueeze(1).float()
            loss = self.criterion(output, target)  # L1 loss for regression applications
        return loss

    def better_training_pipeline(self):
        # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/How_does_adding_dropout_affect_model_performance%3F.ipynb#scrollTo=Z3Cv0XLX54Xo
        with wandb.init(config=self.wandb_config):
            config = wandb.config
            print("train_epochs, config", config)
            # pprint(config)
            wandb.watch(self, self.criterion, log="all", log_freq=1000)

            self.max_epochs = config['epochs']
            self.loss_train_history = []
            self.loss_test_history = []

            test_epoch_avg_loss_prev = 0
            # Loop over epochs
            for epoch in tqdm(range(self.max_epochs)):
                train_epoch_avg_loss_now = self.train_epoch_return_avg_loss()
                test_epoch_avg_loss_now = self.test_epoch_return_avg_loss()

                """ early stopping mechanism """
                stop_early = self.check_early_stopping(test_epoch_avg_loss_now, test_epoch_avg_loss_prev, epoch=epoch)
                if stop_early:
                    print('should stop trainng more epochs now')
                    break
                """ learning rate update """
                # self.decrease_learning_rate(train_epoch_avg_loss_now, test_epoch_avg_loss_prev)

                """ log values """
                wandb.log({"epoch": epoch, "avg_loss_train": train_epoch_avg_loss_now,
                           "avg_loss_test": test_epoch_avg_loss_now,
                           # "example": [wandb.Image(img) for img in imgs],
                           "learning_rate": self.optimizer.defaults['lr']},
                          step=epoch)
                test_epoch_avg_loss_prev = train_epoch_avg_loss_now

            """ model training statistics """
            # self.save_training_model_statistics()
            model_save_path = 'model_training/trained_models/model_' + str(self.max_epochs) + '_' + str(
                self.wandb_config['model']) + '.pth'
            torch.save(self.state_dict(), model_save_path)


    def test_epoch_return_avg_loss(self):
        # from https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/How_does_adding_dropout_affect_model_performance%3F.ipynb#scrollTo=Z3Cv0XLX54Xo
        # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
        self.eval()
        running_loss_test = []

        with torch.no_grad():
            for batch_idx, [imgs, angles, vels] in enumerate(self.train_dataloader, start=0):

                # Load the input features and labels from the test dataset
                imgs, target_angles, target_vels = imgs.to(self.device), angles.to(self.device), vels.to(self.device)

                # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
                output = self.forward(imgs)

                # Compute the loss sum up batch loss
                loss = self.calculate_loss(output, target_angles, target_vels)
                running_loss_test.append(loss.item())
        return np.mean(running_loss_test)

    def train_epoch_return_avg_loss(self):
        # from https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/How_does_adding_dropout_affect_model_performance%3F.ipynb#scrollTo=Z3Cv0XLX54Xo
        # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
        running_loss_train = []
        self.train()

        # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
        for batch_idx, [imgs, angles, vels] in enumerate(self.train_dataloader, start=0):
            # Load the input features and targets from the training dataset
            imgs, target_angles, target_vels = imgs.to(self.device), angles.to(self.device), vels.to(self.device)

            # Reset the gradients to 0 for all learnable weight parameters
            self.optimizer.zero_grad()

            # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
            output = self.forward(imgs)

            # Define our loss function, and compute the loss
            loss = self.calculate_loss(output, target_angles, target_vels)

            # Reset the gradients to 0 for all learnable weight parameters
            self.optimizer.zero_grad()

            # Backward pass: compute the gradients of the loss w.r.t. the model"s parameters
            loss.backward()
            running_loss_train.append(loss.item())

            # Update the neural network weights
            self.optimizer.step()

        return np.mean(running_loss_train)


    def training_pipeline(self):
        with wandb.init(config=self.wandb_config):
            config = wandb.config
            print("train_epochs, config", config)
            # pprint(config)
            no_outputs = self.mlp_head._modules['2'].out_features

            self.max_epochs = config['epochs']
            self.loss_train_history = []
            self.loss_test_history = []
            example_ct_train = 0  # number of examples seen
            batch_ct_train = 0

            example_ct_test = 0  # number of examples seen
            batch_ct_test = 0
            eval_loss_prev = 0  # early stopping

            # total_samples = len(self.train_dataloader.dataset)
            wandb.watch(self, self.criterion, log="all", log_freq=1000)

            # Loop over epochs
            for epoch in tqdm(range(self.max_epochs)):
                running_loss_train = []
                running_loss_test = []

                self.train()
                for i, [imgs, angles, vels] in enumerate(self.train_dataloader):
                    imgs, angles, vels = imgs.to(self.device), angles.to(self.device), vels.to(self.device)

                    # img = imgs[0].cpu().permute(1, 2, 0).numpy()
                    # plt.imshow(img)
                    # plt.show()

                    self.optimizer.zero_grad()
                    output = self.forward(imgs)

                    if no_outputs == 2:
                        target = torch.cat((angles.unsqueeze(1).float(), vels.unsqueeze(1).float()), 1)
                        loss = self.criterion(output, target)  # L1 loss for regression applications
                    else:
                        target = angles.unsqueeze(1).float()
                        loss = self.criterion(output, target)  # L1 loss for regression applications

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss_train.append(loss.item())

                    example_ct_train += len(imgs)
                    batch_ct_train += 1

                self.loss_train_history.append(np.mean(running_loss_train))

                self.eval()
                with torch.no_grad():
                    # self.train()  # put to train for checking if the lower eval test is due to the dropout! - seems like dropout is fking nice
                    for j, [imgs, angles, vels] in enumerate(self.test_dataloader):
                        imgs, angles, vels = imgs.to(self.device), angles.to(self.device), vels.to(self.device)
                        self.optimizer.zero_grad()
                        output = self.forward(imgs)

                        if no_outputs == 2:
                            target = torch.cat((angles.unsqueeze(1).float(), vels.unsqueeze(1).float()), 1)
                            loss = self.criterion(output, target)  # L1 loss for regression applications
                        else:
                            target = angles.unsqueeze(1).float()
                            loss = self.criterion(output, target)  # L1 loss for regression applications

                        running_loss_test.append(loss.item())
                        example_ct_test += len(imgs)
                        batch_ct_test += 1
                    # wandb.log({"loss": np.mean(running_loss_test), "epoch": epoch})
                    eval_loss_now = np.mean(running_loss_test)
                    self.loss_test_history.append(eval_loss_now)

                """ early stopping mechanism """
                stop_early = self.check_early_stopping(eval_loss_now, eval_loss_prev, epoch=epoch)
                eval_loss_prev = eval_loss_now
                if stop_early:
                    print('should stop trainng more epochs now')
                    break
                """ learning rate update """
                self.decrease_learning_rate(eval_loss_now, eval_loss_prev)

                """ log values """
                wandb.log({"epoch": epoch, "avg_loss_train": np.mean(running_loss_train),
                           "avg_loss_test": np.mean(running_loss_test),
                           "example": [wandb.Image(img) for img in imgs],
                           "learning_rate": self.optimizer.defaults['lr']},
                          step=epoch)

        self.save_training_model_statistics()

        model_save_path = 'model_training/trained_models/model_' + str(self.max_epochs) +'_' + str(self.wandb_config['model']) + '.pth'
        torch.save(self.state_dict(), model_save_path)

    def check_early_stopping(self, eval_loss_now, eval_loss_prev, epoch):
        if self.wandb_config['early_stopping']:
            if eval_loss_now > eval_loss_prev:
                self.trigger_times += 1
                # print('self.trigger_times', self.trigger_times, '\teval_loss_now > eval_loss_prev', eval_loss_now > eval_loss_prev)
                if self.trigger_times >= self.patience + 1:  # +1 because of the first eval loss is small
                    model_save_path = 'model_training/trained_models/model_' + str(epoch) + '_' + str(
                        self.wandb_config['model']) + '_' + str(self.max_epochs) + '_early_stopping.pth'
                    torch.save(self.state_dict(), model_save_path)
                    return True
            else:
                self.trigger_times = 0
                # print('self.trigger_times', self.trigger_times, '\teval_loss_now > eval_loss_prev', eval_loss_now > eval_loss_prev)
        return False

    def decrease_learning_rate(self, eval_loss_now, eval_loss_prev):
        if eval_loss_now < eval_loss_prev:
            loss_delta = eval_loss_now - eval_loss_prev
            if loss_delta < 0:
                impr_perc = np.abs(loss_delta)/eval_loss_prev
                # if 0.999*eval_loss_now > eval_loss_prev:
                # decrease lr by 0.1%
                learning_rate_prev = self.optimizer.defaults['lr']
                self.optimizer.defaults['lr'] = (1-impr_perc/10) * learning_rate_prev

    def save_training_model_statistics(self):

        float_formatter = "{:.4f}".format

        final_loss_train = float_formatter(self.loss_train_history[-1])
        min_loss_train = float_formatter(np.min(self.loss_train_history))

        final_loss_test = float_formatter(self.loss_test_history[-1])
        min_loss_test = float_formatter(np.min(self.loss_test_history))

        self.training_model_statistics_dict = dict(final_loss_train=final_loss_train, min_loss_train=min_loss_train,
                                                   final_loss_test=final_loss_test, min_loss_test=min_loss_test)

    def plot_training_history(self):
        plotname = 'model_training/training_plots/model_' + str(self.max_epochs) + '.png'
        f = plt.figure(figsize=(9, 4.8))

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Check losses for overfitting")
        plt.suptitle(str(self.training_model_statistics_dict), fontsize=9)
        plt.plot(self.loss_train_history, label='loss_train_history')
        plt.plot(self.loss_test_history, label='loss_test_history')
        plt.legend()
        plt.savefig(plotname)
        plt.show()

    def evaluate(model, data_loader, loss_history):
        model.eval()

        total_samples = len(data_loader.dataset)
        correct_samples = 0
        total_loss = 0

        with torch.no_grad():
            for data, target in data_loader:
                output = F.log_softmax(model(data), dim=1)
                loss = F.nll_loss(output, target, reduction='sum')
                _, pred = torch.max(output, dim=1)

                total_loss += loss.item()
                correct_samples += pred.eq(target).sum()

        avg_loss = total_loss / total_samples
        loss_history.append(avg_loss)
        print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
              '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
              '{:5}'.format(total_samples) + ' (' +
              '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


if __name__ == "__main__":

    DOWNLOAD_PATH = os.getcwd() + '/data/mnist'
    BATCH_SIZE_TRAIN = 10 # 100
    BATCH_SIZE_TEST = 100 # 1000

    N_EPOCHS = 5 #25

    start_time = time.time()
    model = ViTRegression(image_size=256, patch_size=8, num_outputs=1, channels=1,
                          dim=64, depth=1, heads=2, mlp_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # training function
    data_dir = os.getcwd() + '/data/try_smart_fast_p50'
    model.load_split_train_test(data_dir)

    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)

        # train_epoch(model, optimizer, train_loader, train_loss_history)
        # evaluate(model, test_loader, test_loss_history)

        for inputs, labels in model.trainloader:
            # steps += 1
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            print('labels', labels)
            # translate labels to regression task
            # labels_to_target = dict(east=1, grasp=2, north=3, release=4, south=5, west=6)
            # target = labels_to_target[labels]
            target = labels * 10

            print('target', target)

            optimizer.zero_grad()
            # TODO: Check the sizes to find whats wrong
            output = model.forward(inputs)
            print('output', output)

            loss = F.smooth_l1_loss(output, target.float())  # L1 loss for regression applications
            loss.backward()
            print('loss', loss)
            optimizer.step()

    torch.save(model, 'KEYVIBE_model.pth')
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')