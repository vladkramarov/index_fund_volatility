from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger


def get_callbacks():
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-7, patience=20, verbose=True, mode="min"
    )
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")
    model_summary = ModelSummary()
    return early_stop_callback, lr_logger, logger, checkpoint, model_summary
