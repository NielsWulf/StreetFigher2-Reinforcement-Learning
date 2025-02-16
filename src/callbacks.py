import os
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    """
    Custom callback for training and logging. Saves model checkpoints periodically.
    """
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq  # Frequency of model checkpoint saves
        self.save_path = save_path  # Directory path where checkpoints will be stored

    def _init_callback(self) -> None:
        """
        Create the save directory if it does not already exist.
        """
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)  # Ensure the directory exists

    def _on_step(self) -> bool:
        """
        Called at every step. Saves the model at regular intervals defined by `check_freq`.
        """
        if self.n_calls % self.check_freq == 0:  # Check if it's time to save a checkpoint
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')  # Define checkpoint path
            self.model.save(model_path)  # Save the model
            if self.verbose:
                print(f"Model checkpoint saved at {model_path}")  # Print log message
        return True  # Continue training