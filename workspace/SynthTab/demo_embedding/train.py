from amt_tools.evaluate import validate, append_results, average_results, log_results
from amt_tools import tools

# Regular imports
from tensorboardX import SummaryWriter
from tqdm import tqdm

import os
import random

import numpy as np
import torch

__all__ = [
    'train'
]


def _collect_random_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': None,
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _save_training_state(model, optimizer, scheduler, log_dir, epoch, next_epoch, config_snapshot=None):
    model_iter = int(getattr(model, 'iter', 0))
    payload = {
        'checkpoint_type': 'tabcnn_synthtab_training_state',
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': int(epoch),
        'next_epoch': int(next_epoch),
        'model_iter': model_iter,
        'random_state': _collect_random_state(),
        'config': config_snapshot,
    }
    torch.save(payload, os.path.join(log_dir, f'training-state-{model_iter}.{tools.PYT_EXT}'))


def _save_legacy_state(model, optimizer, log_dir):
    model_iter = int(getattr(model, 'iter', 0))
    torch.save(model, os.path.join(log_dir, f'{tools.PYT_MODEL}-{model_iter}.{tools.PYT_EXT}'))
    torch.save(optimizer.state_dict(), os.path.join(log_dir, f'{tools.PYT_STATE}-{model_iter}.{tools.PYT_EXT}'))


def train(model, train_loader, optimizer, epochs, checkpoints=0, log_dir='.',
          scheduler=None, val_set=None, estimator=None, evaluator=None,
          start_epoch=0, sanity_steps=None, save_full_checkpoints=True,
          config_snapshot=None):
    """
    Implements the training loop for an experiment.

    Parameters
    ----------
    model : TranscriptionModel
      Model to train
    train_loader : DataLoader
      PyTorch Dataloader object for retrieving batches of data
    optimizer : Optimizer
      PyTorch Optimizer for updating weights
    epochs : int
      Number of loops through the dataset;
      Each loop contains a snippet of each track exactly once
    checkpoints : int
      Number of batches in between checkpoints
    log_dir : str
      Path to directory for saving model, optimizer state, and events
    scheduler : Scheduler or None (optional)
      PyTorch Scheduler used to update learning rate
    val_set : TranscriptionDataset or None (optional)
      Dataset to use for validation loops
    estimator : Estimator
      Estimation protocol to use during validation
    evaluator : Evaluator
      Evaluation protocol to use during validation
    start_epoch : int
      Epoch to start from when resuming a full training-state checkpoint
    sanity_steps : int or None
      Optional total number of batches to run before stopping early for smoke tests
    save_full_checkpoints : bool
      Save resumable training-state checkpoints in addition to legacy model/optimizer files
    config_snapshot : dict or None
      Serialized run config stored inside full training-state checkpoints

    Returns
    ----------
    model : TranscriptionModel
      Trained model
    """

    os.makedirs(log_dir, exist_ok=True)

    # Initialize a writer to log any reported results
    writer = SummaryWriter(log_dir)

    # Make sure the model is in training mode
    model.train()

    steps_this_run = 0

    for epoch in tqdm(range(start_epoch, epochs)):
        # Collection of losses for each batch in the loop
        train_loss = dict()
        # Loop through the dataset
        for batch in tqdm(train_loader, desc='Step'):
            # Zero the accumulated gradients
            optimizer.zero_grad()
            # Get the predictions and loss for the batch
            preds = model.run_on_batch(batch)
            # Extract the loss from the output
            batch_loss = preds[tools.KEY_LOSS]
            # Compute gradients based on total loss
            batch_loss[tools.KEY_LOSS_TOTAL].backward()
            # Add all of the losses to the collection
            train_loss = append_results(train_loss, tools.dict_to_array(batch_loss))
            # Perform an optimization step
            optimizer.step()

            # Average the loss from all of the batches within this loop
            train_loss = average_results(train_loss)
            # Log the training loss(es)
            log_results(train_loss, writer, step=model.iter, tag=f'{tools.TRAIN}/{tools.KEY_LOSS}')

            # Increase the iteration count by one
            model.iter += 1
            steps_this_run += 1

            if checkpoints and model.iter % checkpoints == 0:
                _save_legacy_state(model, optimizer, log_dir)
                if save_full_checkpoints:
                    _save_training_state(
                        model, optimizer, scheduler, log_dir,
                        epoch=epoch, next_epoch=epoch, config_snapshot=config_snapshot
                    )

                if val_set is not None and evaluator is not None:
                    # Validate the current model weights
                    validate(model, val_set, evaluator, estimator)
                    # Average the results, log them, and reset the tracking
                    evaluator.finalize(writer, model.iter)
                    # Make sure the model is back in training mode
                    model.train()

            if sanity_steps is not None and steps_this_run >= sanity_steps:
                if save_full_checkpoints:
                    _save_training_state(
                        model, optimizer, scheduler, log_dir,
                        epoch=epoch, next_epoch=epoch, config_snapshot=config_snapshot
                    )
                _save_legacy_state(model, optimizer, log_dir)
                writer.close()
                return model

        if scheduler is not None:
            # Perform a learning rate scheduler step
            scheduler.step()

        if save_full_checkpoints:
            _save_training_state(
                model, optimizer, scheduler, log_dir,
                epoch=epoch, next_epoch=epoch + 1, config_snapshot=config_snapshot
            )

    # Save the final model and optimizer state
    _save_legacy_state(model, optimizer, log_dir)
    if save_full_checkpoints:
        _save_training_state(
            model, optimizer, scheduler, log_dir,
            epoch=max(start_epoch, epochs) - 1, next_epoch=epochs, config_snapshot=config_snapshot
        )

    if val_set is not None and evaluator is not None:
        # Validate the current model weights
        validate(model, val_set, evaluator, estimator)
        # Average the results, log them, and reset the tracking
        evaluator.finalize(writer, model.iter)

    writer.close()
    return model
