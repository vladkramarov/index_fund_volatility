from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

def get_callbacks():
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-6, patience=10, verbose=True, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")
    return early_stop_callback, lr_logger, logger, checkpoint