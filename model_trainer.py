import os
from argparse import Namespace
from copy import deepcopy
from time import time

import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import random_split, DataLoader

import models.modelio
from data import CustomDataset, ImageDataset
from losses.chamfer_loss import ChamferLoss
from losses.dgssm_loss import DGSSMLoss
from losses.mesh_loss import RegularizedMeshLoss


class ModelTrainer:
    def __init__(self, model: models.modelio.LoadableModel, ds: CustomDataset, loss_function, out_dir: str,
                 device: str, args: Namespace):

        self.model = model
        self.ds = ds
        self.batch_size = args.batch
        self.device = device
        self.epochs = args.epochs
        self.out_dir = out_dir
        self.show = args.show
        if 'head_schedule' in args.__dict__:
            self.head_schedule = args.head_schedule

            # deactivate all heads (will be re-activated based on the schedule from args)
            for name in model.dgcnn.head_active:
                model.set_head_active(name, False)

        else:
            self.head_schedule = None

        self.initial_epoch = 0

        # specify frequency of checkpoint saves (after this many epochs)
        self.checkpoint_every = 50

        # setup optimization
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        min_lr = args.lr * 0.05
        if args.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8,
                                                                        patience=math.ceil(0.05*self.epochs),
                                                                        threshold=1e-4, cooldown=math.ceil(0.05*self.epochs),
                                                                        verbose=True, min_lr=min_lr)
        elif args.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=min_lr, verbose=True)
        elif args.scheduler == 'none':
            self.scheduler = None
        else:
            raise ValueError(f'Scheduler "{args.scheduler}" undefined.')

        # loss function
        self.loss_function = loss_function
        self.autocast_enabled = not isinstance(loss_function, (ChamferLoss, RegularizedMeshLoss, DGSSMLoss))

        # create data loaders
        self.validation_split = 0.2  # percentage of the training data being used for validation during training
        val_split = int(len(self.ds) * 0.2)
        ds, valid_ds = random_split(self.ds, lengths=[len(self.ds) - val_split, val_split])
        num_workers = 4 if isinstance(ds, ImageDataset) else 0
        drop_last = len(ds) // 2 >= self.batch_size
        self.train_dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True,  # more efficient data loading
                                   collate_fn=self.ds.get_batch_collate_fn(), drop_last=drop_last)
        self.valid_dl = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True,  # more efficient data loading
                                   collate_fn=self.ds.get_batch_collate_fn())

        # automatic mixed precision
        self.scaler = GradScaler(enabled=self.autocast_enabled)

        # history
        self.training_history = {}
        self.validation_history = {}
        self.best_model = None
        self.best_epoch = 0

        # timer
        self.training_start = 0
        self.epoch_start = 0

    def run(self, initial_epoch=0):
        self.initial_epoch = initial_epoch

        self.initialize()

        epochs = torch.arange(self.initial_epoch, self.epochs)
        for ep, epoch in enumerate(epochs):
            self.epoch_start = time()

            self.model.train()

            if self.head_schedule is not None:
                for name, activation_epoch in self.head_schedule.items():
                    if epoch >= activation_epoch and not self.model.dgcnn.head_active[name]:
                        print(f'activating {name} head')
                        self.model.set_head_active(name)
                print(f'active heads: {self.model.dgcnn.head_active}')

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
            self.training_history[term] = torch.zeros(self.epochs - self.initial_epoch, device=self.device)
            self.validation_history[term] = torch.zeros(self.epochs - self.initial_epoch, device=self.device)

    def forward_step(self, x, y, ep, train):
        # TODO: support additional validation metrics

        with autocast(enabled=self.autocast_enabled):
            x = x.to(self.device)

            # forward pass
            output = self.model(x)

            # UGLY SPECIAL CASE
            if isinstance(self.loss_function, DGSSMLoss):
                # use optimal SSM weights for supervision
                shape = y[0].to(self.device)
                with torch.no_grad():
                    target_weights = self.model.ssm(shape)
                y = (shape, target_weights, y[1].to(self.device))  # shape, target weights and target affine params
            else:
                y = y.to(self.device)

            # loss computation
            loss = self.loss_function(output, y)

        if isinstance(loss, tuple):
            loss, components = loss
        else:
            components = {}

        if train:
            # optimization with mixed precision
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()

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
        history['total_loss'][ep] += loss.detach() * batch_factor
        for term in components.keys():
            history[term][ep] += components[term].detach() * batch_factor

    def after_epoch(self, epoch):
        ep = epoch - self.initial_epoch
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(self.validation_history['total_loss'][ep])
        elif self.scheduler is not None:
            self.scheduler.step()

        # status output
        print(f'\nEPOCH {epoch} ({time() - self.epoch_start:.4f} seconds)')
        print('\t[train]', ' -- '.join(f'{key}: {self.training_history[key][ep]:.4f}' for key in self.training_history.keys()))
        print('\t[valid]', ' -- '.join(f'{key}: {self.validation_history[key][ep]:.4f}' for key in self.validation_history.keys()))
        if self.head_schedule is not None:
            print(f'\tactive heads: {", ".join(name for name in self.model.dgcnn.head_active if self.model.dgcnn.head_active[name])}')

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

        # save best model
        model_path = os.path.join(self.out_dir, 'model.pth')
        print(f'Saving best model from epoch {self.best_epoch} to path "{model_path}"')
        self.model.load_state_dict(self.best_model)
        self.model.save(model_path)

        # training plot
        fig_width, fig_height = plt.rcParams.get('figure.figsize')
        fig, ax = plt.subplots(len(self.training_history.keys()), 1,
                               figsize=(fig_width, len(self.training_history.keys()) * fig_height))
        if len(self.training_history.keys()) == 1:
            ax = [ax]
        fig.suptitle(f'Training Progression. Best model from epoch {self.best_epoch}.')
        for i, key in enumerate(self.training_history.keys()):
            ax[i].plot(np.arange(self.initial_epoch, self.epochs), self.training_history[key].cpu(), label=f'training', c='b')
            ax[i].plot(np.arange(self.initial_epoch, self.epochs), self.validation_history[key].cpu(), label=f'validation', c='r')
            ax[i].set_ylabel(key)
            ax[i].legend()
        ax[-1].set_xlabel('epoch')
        fig.savefig(os.path.join(self.out_dir, f"training_progression.png"), dpi=300)
        if self.show:
            plt.show()
        else:
            plt.close()
