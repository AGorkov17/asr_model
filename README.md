# Keyword Spotting

[Kaggle Competition](https://www.kaggle.com/competitions/keyword-spotting-mipt-2026)

constraints:
* $\le 10^4$ params
* $\le 10^6$ multiply-accumulate operations per 1 second

## Baseline

```bash
cd kws
python3 -m venv kws_venv
. kws_venv/bin/activate
pip install -r requirements.txt
# train
python run.py ++train_dataloader.dataset.manifest_path=<train_manifest> ++val_dataloader.dataset.manifest_path=<val_manifest> ++predict_dataloader.dataset.manifest_path=<test_manifest>
# monitor learning curves in tensorboard
tensorboard --logdir ./lightning_logs
# export to ONNX
python to_onnx.py ++init_weights=<path_to_model>
# visualize graph with netron
netron ./data/kws.onnx
# form submit for Kaggle
python submit.py ++init_weights=<path_to_model> ++train_dataloader.dataset.manifest_path=<train_manifest> ++val_dataloader.dataset.manifest_path=<val_manifest> ++predict_dataloader.dataset.manifest_path=<test_manifest>
```

## TinyTCResNet

`conf/tcresnet.yaml` defines a stronger residual temporal model with waveform
augmentation, SpecAugment, AdamW, cosine learning-rate decay, best-checkpoint
saving, early stopping and both TensorBoard and CSV logs.

```bash
# one-batch smoke test
python run.py --config-name=tcresnet \
  ++train_dataloader.dataset.manifest_path=<train_manifest> \
  ++val_dataloader.dataset.manifest_path=<val_manifest> \
  ++predict_dataloader.dataset.manifest_path=<test_manifest> \
  ++trainer.fast_dev_run=true

# full training
python run.py --config-name=tcresnet \
  ++train_dataloader.dataset.manifest_path=<train_manifest> \
  ++val_dataloader.dataset.manifest_path=<val_manifest> \
  ++predict_dataloader.dataset.manifest_path=<test_manifest> \
  ++trainer.accelerator=gpu ++trainer.devices=1

# submission from the checkpoint printed at the end of training
python submit.py --config-name=tcresnet \
  ++init_weights=<best_checkpoint> \
  ++train_dataloader.dataset.manifest_path=<train_manifest> \
  ++val_dataloader.dataset.manifest_path=<val_manifest> \
  ++predict_dataloader.dataset.manifest_path=<test_manifest> \
  ++trainer.accelerator=gpu ++trainer.devices=1
```

Training artifacts are written under `artifacts/`. In particular, the best
checkpoint is in `artifacts/tiny_tcresnet_v1/checkpoints/`, and the portable
metrics file is under `artifacts/logs/tiny_tcresnet_v1_csv/`.
