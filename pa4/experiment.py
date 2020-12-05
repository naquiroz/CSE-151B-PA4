################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .caption_utils import *
from .constants import ROOT_STATS_DIR
from .dataset_factory import get_datasets
from .file_utils import *
from .model_factory import get_model


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        # Load .json parameters into config_data
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        (
            self.__coco_test,
            self.__vocab,
            self.__train_loader,
            self.__val_loader,
            self.__test_loader,
        ) = get_datasets(config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = (
            None  # Save your best model in this field and use this in test method.
        )
        self.__best_model_state = None

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        # TODO: Make sure Criterion and Optimizer work correctly
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), config_data['experiment']["learning_rate"])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            print("Loading")
            self.__training_losses = read_file_in_dir(
                self.__experiment_dir, 'training_losses.txt'
            )
            self.__val_losses = read_file_in_dir(
                self.__experiment_dir, 'val_losses.txt'
            )
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(
                os.path.join(self.__experiment_dir, 'latest_model.pt')
            )
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(
            start_epoch, self.__epochs
        ):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    def __one_hot_to_number(self, word):
        return self.__vocab(word.argmax().item())

    def __debug(self, captions):
        first_five = captions[:5]
        mapped = [[self.__vocab(self.__one_hot_to_number(word)) for word in caption] for caption in first_five]
        
    def __forward(self, train: bool = False):
        if train:
            self.__model.train()
            loader = self.__train_loader
        else:
            self.__model.eval()
            loader = self.__val_loader

        device = self.device
        vocab_size = len(self.__vocab)
        size = len(loader)

        total_loss = 0

        for i, (images, captions, _) in enumerate(loader):

            images = images.to(device)
            captions = captions.to(device)

            if train:
                self.__optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                output = self.__model(images, captions)
                ####### DEBUGGING
                self.__debug(output)
                ####### DEBUGGING
                loss = self.__criterion(output.view(-1, vocab_size), captions.view(-1))

                total_loss += loss.item() / size
        
                if train:
                    loss.backward()
                    self.__optimizer.step()
        state_dict = self.__model.state_dict()

        losses = self.__training_losses if train else self.__val_losses
        losses.append(total_loss)

        return total_loss


    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        return self.__forward(train=True)

    
    
    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        loss = self.__forward(train=False)
        if len(self.__val_losses) < 2 or loss < self.__val_losses[-2]:
            self.__best_model_state = self.__model.state_dict()
        return loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model = self.__model.load_state_dict(self.__best_model_state)
        self.__model.eval()

        device = self.device
        test_loss = 0
        bleu1_score = 0
        bleu4_score = 0

        with torch.no_grad():
            for i, (images, captions, img_ids) in enumerate(self.__test_loader):
                size = len(images)
                images = images.to(device)
                captions = captions.to(device)
                prediction = self.__model.generate_captions(images).to(device)
                test_loss += self.__criterion(
                    prediction.reshape(-1, prediction.shape[2]), captions.reshape(-1)
                ) / size
                bleu1_score += bleu1(captions, prediction) / size
                bleu4_score += bleu4(captions, prediction) / size

        result_str = (
            "Test Performance: Loss: {}, Perplexity: {}, Bleu1: {}, Bleu4: {}".format(
                test_loss, bleu1_score, bleu4_score
            )
        )
        self.__log(result_str)

        return test_loss, bleu1_score, bleu4_score

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(
            self.__experiment_dir, 'training_losses.txt', self.__training_losses
        )
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(
            self.__current_epoch + 1,
            train_loss,
            val_loss,
            str(time_elapsed),
            str(time_to_completion),
        )
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
