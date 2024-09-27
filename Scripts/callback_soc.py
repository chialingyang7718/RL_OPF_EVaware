from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import os


class SOCCallback(BaseCallback):
    def __init__(self, log_dir):
        super(SOCCallback, self).__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        # Use env_method to call the get_soc_percentages method across all environments
        all_soc_percentages = self.training_env.env_method("get_soc")

        # Iterate through each set of soc_percentages returned by the environments
        for idx, soc_percentages in enumerate(all_soc_percentages):
            # Log each SOC value individually or aggregate them
            for i, soc in enumerate(soc_percentages):
                self.writer.add_scalar(
                    f"SOC/Environment_{idx}_Storage_{i}", soc, self.num_timesteps
                )
        return True

    def _on_training_end(self) -> None:
        self.writer.close()
