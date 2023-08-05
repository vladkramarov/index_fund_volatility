import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer
import data_processing.dataset_and_loaders as dataset_and_loaders
from pytorch_forecasting.metrics import QuantileLoss
import training.callbacks as callbacks
import training.config
from pytorch_forecasting.metrics import RMSE    
training_dataset, validation_dataset, train_dataloader, val_dataloader = dataset_and_loaders.get_timeseries_datasets_and_dataloaders()
early_stop_callback, lr_logger, logger = callbacks.get_callbacks()

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="cpu",
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.15,
    overfit_batches=1
)

tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate= 0.001,
        hidden_size= training.config.HIDDEN_SIZE, #13,
        attention_head_size= training.config.ATTENTION_HEADS, #6,
        dropout= training.config.DROPOUT,
        hidden_continuous_size= training.config.HIDDEN_CONTINUOUS_SIZE,#11,
        output_size = 1, #7,
        loss=RMSE(),
)

#overfit the model on a single batch

trainer.fit(tft, train_dataloader, val_dataloaders=val_dataloader)
# preds = trainer.validate(tft, val_dataloader, ckpt_path=trainer.checkpoint_callback.best_model_path)
