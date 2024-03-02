import logging


from .data_generate import build_voc_semi_loader, build_vocloader,build_voc_semi_loader_new

logger = logging.getLogger("global")


def get_loader(cfg, seed=0):
    cfg_dataset = cfg["dataset"]



    if cfg_dataset["type"] == "our":
        train_loader_sup, train_loader_unsup = build_voc_semi_loader( "train", cfg, seed=seed )
        val_loader = build_vocloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader


    else:
        raise NotImplementedError(
            "dataset type {} is not supported".format(cfg_dataset)
        )


def get_loader_new(cfg, seed=0):
    cfg_dataset = cfg["dataset"]



    if cfg_dataset["type"] == "our":
        train_loader_sup= build_voc_semi_loader_new( "train", cfg, seed=seed )
        val_loader = build_vocloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    else:
        raise NotImplementedError(
            "dataset type {} is not supported".format(cfg_dataset)
        )