import os
from copy import deepcopy
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader

import models.modelio


class ModelTrainer:
    def __init__(self, model: models.modelio.LoadableModel, train_ds: Dataset, loss_function, learning_rate: float, batch_size: int, device: str,
                 epochs: int, out_dir: str, show: bool):

        self.model = model
        self.ds = train_ds
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.out_dir = out_dir
        self.show = show

        self.initial_epoch = 0

        # specify frequency of checkpoint saves (after this many epochs)
        self.checkpoint_every = 50

        # setup optimization
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=50,
                                                                    threshold=1e-4, cooldown=50, verbose=True)

        # loss function
        self.loss_function = loss_function

        # create data loaders
        self.validation_split = 0.2  # percentage of the training data being used for validation during training
        val_split = int(len(self.ds) * 0.2)
        train_ds, valid_ds = random_split(self.ds, lengths=[len(self.ds) - val_split, val_split])
        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.valid_dl = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)

        # history
        self.training_history = {}
        self.validation_history = {}
        self.best_model = None
        self.best_epoch = 0

        # timer
        self.training_start = 0

    def run(self, initial_epoch=0):
        self.initial_epoch = initial_epoch

        self.initialize()

        epochs = torch.arange(self.initial_epoch, self.epochs)
        for ep, epoch in enumerate(epochs):
            self.model.train()
            for x_batch, y_batch in self.train_dl:
                self.forward_step(x_batch, y_batch, ep, train=True)

            self.model.eval()
            for x_batch, y_batch in self.valid_dl:
                with torch.no_grad():
                    self.forward_step(x_batch, y_batch, ep, train=False)

            self.after_epoch(epoch.item())

        self.finalize()

    def initialize(self):
        print('\nTRAINING MODEL ...\n')
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)

        self.training_start = time()
        self.model.to(self.device)

    def init_history(self, loss_term_labels):
        for term in loss_term_labels:
            self.training_history[term] = torch.zeros(self.epochs - self.initial_epoch)
            self.validation_history[term] = torch.zeros(self.epochs - self.initial_epoch)

    def forward_step(self, x, y, ep, train):
        # TODO: support additional validation metrics

        # forward pass
        output = self.model(x.to(self.device))

        # loss computation
        loss = self.loss_function(output, y.to(self.device))
        if isinstance(loss, tuple):
            loss, components = loss
        else:
            components = {}

        if train:
            # optimization
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            dl = self.train_dl
            history = self.training_history
        else:
            dl = self.valid_dl
            history = self.validation_history

        # init history dicts if they are empty
        if history == {}:
            self.init_history(loss_term_labels=['total_loss'] + list(components.keys()))

        # save mean batch statistics (weighting based on number of samples, works with or without drop_last in dl)
        batch_factor = len(x) / (len(dl.dataset) if not dl.drop_last else len(dl) * self.batch_size)
        history['total_loss'][ep] += loss * batch_factor
        for term in components.keys():
            history[term][ep] += components[term] * batch_factor

    def after_epoch(self, epoch):
        ep = epoch - self.initial_epoch
        self.scheduler.step(self.validation_history['total_loss'][ep])

        # status output
        print(f'\nEPOCH {epoch}:')
        print('Training Metrics:')
        print(' - '.join(f'\t{key}: {self.training_history[key][ep]:.4f}' for key in self.training_history.keys()))
        print('Validation Metrics:')
        print(' - '.join(f'\t{key}: {self.validation_history[key][ep]:.4f}' for key in self.validation_history.keys()))

        # save best snapshot  # TODO: allow to specify criterion for best model
        if self.validation_history['total_loss'][ep] <= self.validation_history['total_loss'][self.best_epoch]:
            self.best_model = deepcopy(self.model.state_dict())
            self.best_epoch = ep

        # save checkpoint
        if (epoch + 1) % self.checkpoint_every == 0:
            self.model.save(os.path.join(self.out_dir, 'checkpoints', f'{epoch}.pth'))

        # TODO: add visualization possibility

    def finalize(self):
        # stop the timer
        total_train_time_in_s = time() - self.training_start
        print(f'\nDone. Took {total_train_time_in_s / 60:.4f} min')

        # training plot
        plt.figure()
        plt.title(f'Training Progression. Best model from epoch {self.best_epoch}.')
        for key, values in self.training_history.items():
            plt.plot(np.arange(self.initial_epoch, self.epochs), values, label=f'training {key}')
        for key, values in self.validation_history.items():
            plt.plot(np.arange(self.initial_epoch, self.epochs), values, label=f'validation {key}')
        plt.legend()
        plt.savefig(os.path.join(self.out_dir, f"training_progression.png"), dpi=300)
        if self.show:
            plt.show()
        else:
            plt.close()

        # save best model
        model_path = os.path.join(self.out_dir, 'model.pth')
        print(f'Saving best model from epoch {self.best_epoch} to path "{model_path}"')
        self.model.load_state_dict(self.best_model)
        self.model.save(model_path)
