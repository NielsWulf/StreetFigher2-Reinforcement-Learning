import os
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    """
    Custom callback for training and logging. Saves model checkpoints periodically.
    """
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        """
        Create the save directory if it does not already exist.
        """
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at every step. Saves the model at regular intervals defined by `check_freq`.
        """
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
            if self.verbose:
                print(f"Model checkpoint saved at {model_path}")
        return True
