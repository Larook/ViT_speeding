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

from SimulationData import SimulationData
from SimulationImageDataset import SimulationImageDataset
from model_training import vit_pytorch


class ResnetRegression(nn.Module):
    def __init__(self, wandb_config, num_outputs):
        self.num_outputs = num_outputs
        torch.manual_seed(42)
        super().__init__()
        self.wandb_config = wandb_config
        print('self.wandb_config', self.wandb_config)
        pprint.pprint(self.wandb_config)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(pretrained=True).to(self.device)

        # apparently cannot really use resnet18 for pretraining without changing the last - fully connected layer
        self.model.fc = nn.Linear(512, self.num_outputs).to(self.device)

        self.optimizer = self.build_optimizer(self.wandb_config['optimizer'])
        self.criterion = F.smooth_l1_loss

    def build_optimizer(self, optimizer):
        if optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.wandb_config['learning_rate'], momentum=0.9)
        elif optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.wandb_config['learning_rate'])
        return optimizer

    def load_dataloaders(self, pickle_df_path):
        """ reads the pickles from the directory and updates dfs, SimulationData and dataloaders """
        # load training data
        self.data = SimulationData(create=False)
        self.train_df, self.test_df = self.data.load_dfs_from_pickles(create=False, training_percentage=0.8,
                                                                      pickle_df_path=pickle_df_path, shuffle=True)

        train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                               # transforms.Grayscale(num_output_channels=1),
                                               transforms.ColorJitter(brightness=.5, hue=.3),
                                               transforms.RandomEqualize(),
                                               # transforms.RandomVerticalFlip(),
                                               transforms.ToTensor()
                                               ])

        dataset_train = SimulationImageDataset(main_df=self.train_df, transform=train_transforms)
        dataset_test = SimulationImageDataset(main_df=self.test_df, transform=train_transforms)

        # create dataloaders
        self.train_dataloader = DataLoader(dataset_train, batch_size=self.wandb_config['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(dataset_test, batch_size=self.wandb_config['batch_size'], shuffle=True)
        pass

    def forward(self, img, mask=None):
        return self.model(img.to(self.device))

    def train_epochs(self):
        with wandb.init(config=self.wandb_config):
            config = wandb.config
            print("train_epochs, config", config)
            # pprint(config)

            self.max_epochs = config['epochs']
            self.loss_train_history = []
            self.loss_test_history = []
            example_ct_train = 0  # number of examples seen
            batch_ct_train = 0

            example_ct_test = 0  # number of examples seen
            batch_ct_test = 0

            # total_samples = len(self.train_dataloader.dataset)
            wandb.watch(self, self.criterion, log="all", log_freq=1000)

            # Loop over epochs
            for epoch in tqdm(range(self.max_epochs)):
                running_loss_train = []
                running_loss_test = []

                self.train()
                for i, [imgs, angles, vels] in enumerate(self.train_dataloader):
                    imgs, angles, vels = imgs.to(self.device), angles.to(self.device), vels.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model.forward(imgs)

                    no_outputs = self.num_outputs
                    if no_outputs == 2:
                        target = torch.cat((angles.unsqueeze(1).float(), vels.unsqueeze(1).float()), 1)
                        loss = self.criterion(output, target)  # L1 loss for regression applications
                    else:
                        target = angles.unsqueeze(1).float()
                        loss = self.criterion(output, target)  # L1 loss for regression applications
                    loss.backward()
                    self.optimizer.step()
                    running_loss_train.append(loss.item())

                    example_ct_train += len(imgs)
                    batch_ct_train += 1

                self.loss_train_history.append(np.mean(running_loss_train))

                self.eval()
                for j, [imgs, angles, vels] in enumerate(self.test_dataloader):
                    imgs, angles, vels = imgs.to(self.device), angles.to(self.device), vels.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model.forward(imgs)
                    target = angles.unsqueeze(1).float()
                    loss = self.criterion(output, target)  # L1 loss for regression applications
                    running_loss_test.append(loss.item())
                    example_ct_test += len(imgs)
                    batch_ct_test += 1

                wandb.log({"epoch": epoch, "avg_loss_train": np.mean(running_loss_train),
                           "avg_loss_test": np.mean(running_loss_test),
                           "example": [wandb.Image(img) for img in imgs]},
                          step=epoch)

                # wandb.log({"loss": np.mean(running_loss_test), "epoch": epoch})
                self.loss_test_history.append(np.mean(running_loss_test))

        self.save_training_model_statistics()

        model_save_path = 'model_training/trained_models/model_' + str(self.max_epochs) +'_' + str(self.wandb_config['model']) + '.pth'
        torch.save(self.state_dict(), model_save_path)

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