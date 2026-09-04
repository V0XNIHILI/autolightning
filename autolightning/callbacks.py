import os
from pytorch_lightning.callbacks import Callback

class SaveInitialCheckpoint(Callback):
    def on_train_start(self, trainer, pl_module):
        path = os.path.join(trainer.default_root_dir, "checkpoints", "initial.ckpt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        trainer.save_checkpoint(path)
