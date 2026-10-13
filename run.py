import hydra
import omegaconf
from pytorch_lightning import seed_everything

from src.module import KWS
from utils import omegaconf_extension


@omegaconf_extension
@hydra.main(version_base="1.2", config_path="conf", config_name="conv1d.yaml")
def main(conf: omegaconf.DictConfig) -> None:
    seed_everything(314, workers=True)
    module = KWS(conf)
    loggers = [hydra.utils.instantiate(conf.logger)]
    if conf.get("csv_logger") is not None:
        loggers.append(hydra.utils.instantiate(conf.csv_logger))

    callbacks = []
    if conf.get("checkpoint") is not None:
        callbacks.append(hydra.utils.instantiate(conf.checkpoint))
    if conf.get("early_stopping") is not None:
        callbacks.append(hydra.utils.instantiate(conf.early_stopping))

    trainer = hydra.utils.instantiate(
        conf.trainer,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(module)

    for callback in callbacks:
        best_model_path = getattr(callback, "best_model_path", None)
        if best_model_path:
            print(f"Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
