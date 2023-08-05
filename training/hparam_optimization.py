from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import data_processing.dataset_and_loaders as dataset_and_loaders
import pickle
import training.config
import training.callbacks as tr_callbacks
from lightning.pytorch.tuner import Tuner
import training.training as tr_training
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
from pytorch_forecasting.metrics import SMAPE, MAPE, MASE, RMSE, QuantileLoss
import training.config
import datetime

train_dataset, val_dataset, train_dataloader, val_dataloader = dataset_and_loaders.get_timeseries_datasets_and_dataloaders()
early_callback, _, logger = tr_callbacks.get_callbacks()

def learn_rate_tuner(train_dataset: TimeSeriesDataSet = train_dataset, 
                    train_dataloader: DataLoader = train_dataloader, 
                    val_dataloader: DataLoader = val_dataloader):
    
    trainer = tr_training.get_trainer()
    tft = tr_training.get_tft_model(train_dataset)
    tuner = Tuner(trainer).lr_find(
        tft, train_dataloader, val_dataloader, min_lr=1e-6, max_lr=0.5, early_stop_threshold=None)
    
    fig = tuner.plot(show=True, suggest=True)
    fig.show()
    return tuner

def hparam_optimization():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=70,
        max_epochs=20,
        timeout=3600*18,
        gradient_clip_val_range=(0.05, 0.50),
        output_size = 3,
        hidden_size_range=(18, 48),
        hidden_continuous_size_range=(18, 48),
        attention_head_size_range=(2, 6),
        learning_rate = 0.0000069843,
        dropout_range=(0.1, 0.5),
        trainer_kwargs=dict(limit_train_batches=50),
        log_dir="/Users/vladyslavkramarov/Documents/stock_volatility/lightning_logs/optuna_quantile_loss_63_input_10_output",
        use_learning_rate_finder=False, loss=QuantileLoss([0.1, 0.5, 0.9]))

    with open(f"test_quantile_loss_{time_now}", "wb") as f:
        pickle.dump(study, f)

    print(study.best_trial)
    return study

if __name__ == "__main__":
    study = hparam_optimization()
    