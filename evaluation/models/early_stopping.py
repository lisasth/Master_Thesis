import numpy as np
from src.models import utils
import os


class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, save_path, patience=10, verbose=False, delta=0, min_iteration=3000, trace_func=print):
        """
        Args:
            save_path(str): The path to the folder where models will be saved.
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation accuracy improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            min_iteration (int): The minimum iteration when it starts to save the model.
                            Default: 3000
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_acc = None
        self.early_stop_flag = False
        self.val_acc_max = 0
        self.delta = delta
        self.min_iteration = min_iteration
        self.trace_func = trace_func
        self.save_path = save_path

    def __call__(self, val_acc, model, iteration):
        # Check if iteration has reach the minimum iteration
        if iteration < self.min_iteration:
            self.trace_func(f"Current iteration {iteration} is smaller than the minimum iteration {self.min_iteration}, the model will not be saved now")
            return

        if self.best_val_acc is None:
            self.best_val_acc = val_acc
            self.save_checkpoint(val_acc, model, iteration)
        elif val_acc > self.best_val_acc + self.delta:
            # Significant improvement detected
            self.best_val_acc = val_acc
            self.save_checkpoint(val_acc, model, iteration)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop_flag = True

    def save_checkpoint(self, val_acc, model, iteration):
        '''Saves model when validation accuracy increases.'''
        model_path = os.path.join(self.save_path, f'checkpoint_iteration_{iteration}.pt')
        if self.verbose:
            self.trace_func(f'Validation accuracy increased from {self.val_acc_max:.6f} to {val_acc:.6f}. \n'  
                            f'Saving model to {model_path}')
        utils.torch_save(model, model_path)
        print("checkpoint saved to model_path:", model_path)
        self.val_acc_max = val_acc