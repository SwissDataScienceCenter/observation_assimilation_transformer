import os
from datetime import datetime
import warnings
import importlib

from oat.model_module import model
from oat.data_module.setup import setup_data_and_transform
from oat.data_module.transform import Transform
from oat.predictor_module import OATPredictor
from oat.predictor_module.schedulers import estimate_total_steps
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import logging
from pathlib import Path
import pickle
from typing import Dict, Tuple, List, Any
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.profilers import Profiler
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_oat_cfg(cfg: DictConfig, field_extent: Dict, logger: logging.Logger) -> model.OATConfig | model.UninformedOATConfig:
    model_cfg = cfg.model
    data_cfg = cfg.dataset
    logger.info("Setting up model configs...")
    global_input_chans = len(data_cfg.nwp_vars) + len(data_cfg.topo_vars)
    global_encoder_cfg = model.GlobalEncoderConfig(
        **model_cfg.global_encoder,
        in_channels=global_input_chans,
        field_extent_max=field_extent['max'],
        field_extent_min=field_extent['min']
        )

    topography_encoder_cfg = model.TopographicEncoderConfig(
        **model_cfg.topographic_encoder,
        in_channels=len(data_cfg.topo_vars)
        )

    time_cond_dim = 4 if data_cfg.use_time_embedding else 0
    nwp_topo_fusion_cfg = model.NwpTopoFusionConfig(
        **model_cfg.nwp_topo_fusion,
        cond_dim=time_cond_dim,
        out_channels=model_cfg.d_model)

    if list(cfg.dataset.obs_variable_to_distribution.items()) != list(cfg.model.output_head.variable_to_distribution.items()):
        raise ValueError("Mismatch between observations variable distributions and output head variable distributions (both keys and order must match)")
    if list(cfg.model.output_head.variable_to_distribution.keys()) != cfg.dataset.obs_vars:
        raise ValueError("Variables in output head distribution config do not match observation variables")
    if len(cfg.model.output_head.variable_to_distribution) != len(cfg.dataset.obs_vars):
        raise ValueError("Number of variables in output head distribution config does not match number of observation variables")
    logger.info(
        f"Model will assume the following index to variable mapping: "
        f"{ {idx: var for idx, var in enumerate(cfg.dataset.obs_vars)} }"
        "\n And variable to distribution mapping: "
        f"{ cfg.model.output_head.variable_to_distribution }"
    )
    output_head_cfg = model.OutputHeadConfig(
        in_features=model_cfg.d_model,
        **model_cfg.output_head)

    if get_model_cls(cfg) == model.UninformedOAT:
        logger.info("Initialized uninformed model cfg...")
        return model.UninformedOATConfig(
            global_encoder=global_encoder_cfg,
            topographic_encoder=topography_encoder_cfg,
            nwp_topo_fusion=nwp_topo_fusion_cfg,
            output_head=output_head_cfg,
            num_samples=model_cfg.num_samples
        )

    masking_cfg = model.MaskingConfig(**model_cfg.masking)

    transformer_cfg = model.TransformerConfig(
        d_geo_bias=len(data_cfg.geo_vars),
        d_model=model_cfg.d_model,
        **model_cfg.transformer)

    n_obs_vars = len(data_cfg.obs_vars)
    condition_on_mask = model_cfg.innovation_fusion.condition_on_mask
    innovation_cond_dim = [time_cond_dim, n_obs_vars] if condition_on_mask else time_cond_dim
    innovation_fusion_cfg = model.InnovationFusionConfig(
        **model_cfg.innovation_fusion,
        cond_dim=innovation_cond_dim,
        out_channels=model_cfg.d_model)

    if not model_cfg.get("learnable_gating", False):
        logger.info("Learnable gating disabled.")
        learnable_gating_cfg = None
    else:
        logger.info("Learnable gating enabled.")
        learnable_gating_cfg = model.LearnableGatingConfig(
            **model_cfg.learnable_gating,
            in_channels=n_obs_vars,
            out_channels=model_cfg.d_model)

    oat_config = model.OATConfig(
        global_encoder=global_encoder_cfg,
        topographic_encoder=topography_encoder_cfg,
        nwp_topo_fusion=nwp_topo_fusion_cfg,
        innovation_fusion=innovation_fusion_cfg,
        masking=masking_cfg,
        transformer=transformer_cfg,
        output_head=output_head_cfg,
        learnable_gating=learnable_gating_cfg,
        num_samples=model_cfg.num_samples,
        pre_oat_loss_weight=model_cfg.pre_oat_loss_weight,
        mask_innovations=model_cfg.mask_innovations
    )
    logger.info("Initialized model cfg...")
    return oat_config

def setup_oat_predictor(model_cfg: model.OATConfig | model.UninformedOATConfig,
                    estimated_total_step: int,
                    var_names: str,
                    transforms: tuple[Transform], cfg: DictConfig) -> OATPredictor:

    from importlib import import_module

    # optimizer
    module_path, class_name = cfg.predictor.kwargs.optimizer_class.rsplit(".", 1)
    module = import_module(module_path)
    optim_cls = getattr(module, class_name)

    # lr scheduler
    module_path, class_name = cfg.predictor.kwargs.lr_scheduler_class.rsplit(".", 1)
    module = import_module(module_path)
    sched_cls = getattr(module, class_name)

    # oat loss scheduler
    oat_loss_scheduler_class = cfg.predictor.kwargs.get("oat_loss_scheduler_class", None)
    if oat_loss_scheduler_class is not None:
        module_path, class_name = cfg.predictor.kwargs.oat_loss_scheduler_class.rsplit(".", 1)
        module = import_module(module_path)
        oat_loss_sched_cls = getattr(module, class_name)
        oat_loss_sched_params = cfg.predictor.kwargs.oat_loss_scheduler_params
        if oat_loss_sched_params['total_steps'] is None:
            oat_loss_sched_params['total_steps'] = estimated_total_step
            
            logger.info(f"Config did not specify total_steps for the OAT loss scheduler. It has been set to the (manual) estimated total training steps: {estimated_total_step}\
                \nCheck later logs for a comparison with the steps estimated by the lightning trainer.")
        else:
            logger.info(f"OAT loss scheduler total_steps from config: {oat_loss_sched_params['total_steps']} \
                \nCheck later logs for a comparison with the steps estimated by the lightning trainer.")
            
    else:
        oat_loss_sched_cls = None
        oat_loss_sched_params = None
        
    
    predictor = OATPredictor(
        oat_model_cfg=model_cfg,
        input_transform=transforms[0],
        output_transform=transforms[1],
        output_var_names=var_names,
        lr=cfg.predictor.lr,
        optimizer_class=optim_cls,
        optimizer_params=cfg.predictor.kwargs.optimizer_params,
        lr_scheduler_class=sched_cls,
        lr_scheduler_params=cfg.predictor.kwargs.lr_scheduler_params,
        oat_loss_scheduler_class=oat_loss_sched_cls,
        oat_loss_scheduler_params=oat_loss_sched_params
    )

    return predictor


def setup_gp_predictor(
    cfg: DictConfig,
    field_extent: Dict,
    obs_vars: List[str],
    transforms: tuple[Transform, Transform],
) -> "GPPredictor":
    from importlib import import_module
    from oat.model_module.gp_baseline import GPBaselineConfig, GPBaseline
    from oat.predictor_module.gp_predictor import GPPredictor

    data_cfg = cfg.dataset
    model_cfg = cfg.model

    gp_topography_kwargs = None
    if bool(model_cfg.get("use_topography", False)):
        gp_topography_kwargs = {
            **dict(model_cfg.topographic_encoder),
            "in_channels": len(data_cfg.topo_vars),
        }

    gp_config = GPBaselineConfig(
        obs_vars=obs_vars,
        nwp_vars=list(data_cfg.nwp_vars),
        field_extent=field_extent,
        use_nwp=bool(model_cfg.use_nwp),
        use_time_emb=bool(model_cfg.use_time_emb),
        use_topography=bool(model_cfg.get("use_topography", False)),
        topography_encoder_kwargs=gp_topography_kwargs,
        mean_hidden_dims=tuple(model_cfg.mean_hidden_dims),
        kernel_hidden_dims=tuple(model_cfg.kernel_hidden_dims),
        kernel_embed_dim=int(model_cfg.kernel_embed_dim),
        mean_groups=model_cfg.get("mean_groups"),
        cov_groups=model_cfg.get("cov_groups"),
        noise_variance=float(model_cfg.noise_variance),
        eps=float(model_cfg.get("eps", 1e-3)),
        loss_fn=str(model_cfg.loss_fn),
        num_samples=int(model_cfg.num_samples),
    )
    gp_model = GPBaseline(gp_config)

    module_path, class_name = cfg.predictor.kwargs.optimizer_class.rsplit(".", 1)
    optim_cls = getattr(import_module(module_path), class_name)

    sched_cls = None
    sched_params: dict = {}
    if cfg.predictor.kwargs.get("scheduler_class") is not None:
        module_path, class_name = cfg.predictor.kwargs.scheduler_class.rsplit(".", 1)
        sched_cls = getattr(import_module(module_path), class_name)
        sched_params = dict(cfg.predictor.kwargs.get("scheduler_params", {}))

    return GPPredictor(
        gp_model=gp_model,
        input_transform=transforms[0],
        output_transform=transforms[1],
        output_var_names=obs_vars,
        lr=float(cfg.predictor.lr),
        optimizer_class=optim_cls,
        optimizer_params=dict(cfg.predictor.kwargs.get("optimizer_params", {})),
        scheduler_class=sched_cls,
        scheduler_params=sched_params,
    )


def setup_profiler(cfg: DictConfig) -> Profiler | None:
    if cfg.profiler.type == "simple":
        dirpath = os.path.join("profiler_outputs", cfg.logging.group)
        
        now_str = datetime.now().strftime("%Y%m%d")
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    
        filename = f"{cfg.logging.run_name_prefix}_{now_str}_job{slurm_job_id}"
        
        return SimpleProfiler(dirpath=dirpath, filename=filename)

    if cfg.profiler.type == "none" or cfg.profiler.type is None:
        return None

    raise ValueError(f"Profiler type {cfg.profiler.type} unknown.")
        
def setup_logger_and_callbacks(cfg: DictConfig) -> Tuple[WandbLogger, List]:
    base_name = cfg.logging.run_name_prefix
    now_str = datetime.now().strftime("%Y%m%d")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = f"{base_name}_{now_str}_job{slurm_job_id}"
    checkpoint_dir = os.path.join("checkpoints", cfg.logging.group, run_name)
    wandb_logger = WandbLogger(name=run_name,
                         project=cfg.logging.project,
                         group=cfg.logging.group,
                         entity=cfg.logging.entity,
                         log_model=False)

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        dirpath=checkpoint_dir,
        filename="best-model-{epoch:02d}-{val_loss:.4f}"
    )
    # NOTE maybe use cfg for patience
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, "config.yaml"))
    return wandb_logger, [checkpoint_callback, early_stop_callback, lr_monitor]


def save_fixtures_for_tests(input_transform: Transform, output_transform: Transform, train_loader: DataLoader, cfg: DictConfig) -> None:
    assert cfg.dataloader.train.batch_size == 1, "Please use batch size = 1 when saving fixtures."

    logger.info("Saving examples of transforms as fixtures ...")
    ROOT = Path(__file__).resolve().parents[1]

    out_file = ROOT / "test" / "oat_test" / "input_transform_fixture.pickle"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(input_transform, f)
    logger.info(f"Saved input_transform object to {out_file}")

    out_file = ROOT / "test" / "oat_test" / "output_transform_fixture.pickle"
    with open(out_file, "wb") as f:
        pickle.dump(output_transform, f)
    logger.info(f"Saved output_transform object to {out_file}")

    batch = next(iter(train_loader))
    out_file = ROOT / "test" / "oat_test" / "example_batch_fixture.pickle"
    with open(out_file, "wb") as f:
        pickle.dump(batch, f)
    logger.info(f"Saved batch fixture object to {out_file}")



def get_model_cls(cfg: DictConfig) -> Any:
    model_module_str, model_cls_str = cfg.model.model_cls.rsplit(".", 1)
    model_module = importlib.import_module(model_module_str)
    return getattr(model_module, model_cls_str)

@hydra.main(config_path="../configs/", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # set global seed
    OmegaConf.resolve(cfg) # for referenced fields
    torch.manual_seed(cfg.seed)
    model_type = HydraConfig.get().runtime.choices.get("model", "oat")

    # Get model class
    model_cls = get_model_cls(cfg)
    # Check that only "informed" models have a train_split_fraction > 0.0
    # and force it to 0.0 otherwise.
    if not model_cls.requires_observation and cfg.split.train_split_fraction > 0.0:
        warnings.warn(f"Observation fraction = {cfg.split.train_split_fraction} is incompatible with \
            an uninformed model ({model_cls}). It will be forcibly overridden to 0.0")
        cfg.split.train_split_fraction = 0.0
        
    dataset, len_train_dataset, train_loader, val_loader, test_loader, input_transform, output_transform = setup_data_and_transform(cfg, logger)
    
    if cfg.get("save_fixtures", False):
        save_fixtures_for_tests(input_transform=input_transform,
                                output_transform=output_transform,
                                train_loader=train_loader,
                                cfg=cfg)
        return

    field_extent = {k: v.tolist() for k, v in dataset.nwp_extent.items()}
    wandb_logger, callbacks = setup_logger_and_callbacks(cfg)
    
    # print config
    logger.info(f"Config :\n {OmegaConf.to_yaml(cfg)}")

    if model_type == "baseline":
        from oat.model_module.baseline import NWPInterpolatorConfig
        from oat.predictor_module.baseline_predictor import create_baseline_predictor

        baseline_config = NWPInterpolatorConfig(
            field_extent=field_extent,
            **cfg.model
        )
        pre_transform = dataset.get_baseline_pre_transform()
        predictor = create_baseline_predictor(
            config=baseline_config,
            obs_vars=list(dataset.config.obs_vars),
            pre_transform=pre_transform,
        )

        trainer = Trainer(**cfg.trainer, logger=wandb_logger)
        trainer.validate(predictor, val_loader)
    elif model_type == "gp_baseline":
        predictor = setup_gp_predictor(
            cfg=cfg,
            field_extent=field_extent,
            obs_vars=list(dataset.config.obs_vars),
            transforms=[input_transform, output_transform],
        )

        profiler = setup_profiler(cfg=cfg)

        trainer = Trainer(
            **cfg.trainer,
            callbacks=callbacks,
            logger=wandb_logger,
            profiler=profiler,
        )
        trainer.fit(
            model=predictor,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        if profiler is not None and trainer.is_global_zero:
            print(profiler.summary())
    else:
        oat_model_cfg = setup_oat_cfg(cfg, field_extent, logger)

        var_names = dataset.config.obs_vars
        
        # Estimate the total number of training steps. Can be used by the OAT loss scheduler if its total_steps is set to None.
        estimated_total_steps = estimate_total_steps(
            dataset_size=len_train_dataset,
            batch_size=cfg.dataloader.train.batch_size,
            max_epochs=cfg.trainer.max_epochs,
            max_steps=OmegaConf.select(cfg.trainer, "max_steps", default=None),
            accumulate_grad_batches=OmegaConf.select(cfg.trainer, "accumulate_grad_batches", default=1),
            num_nodes=OmegaConf.select(cfg.trainer, "num_nodes", default=1),
            num_gpus_per_node=OmegaConf.select(cfg.trainer, "devices", default=4),
            limit_train_batches=OmegaConf.select(cfg.trainer, "limit_train_batches", default=1.0),
            drop_last=OmegaConf.select(cfg.dataloader.train, "drop_last", default=False)
        )
        logger.info(f"Manual estimated total optimizer steps: {estimated_total_steps}")

        predictor = setup_oat_predictor(model_cfg=oat_model_cfg,
                                        estimated_total_step=estimated_total_steps,
                                        var_names=var_names,
                                        transforms=[input_transform, output_transform],
                                        cfg=cfg)


        # NOTE check if compiling model helps
        # predictor = torch.compile(predictor)

        profiler = setup_profiler(cfg=cfg)

        trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=wandb_logger, profiler=profiler)
        trainer.fit(model=predictor,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

        if profiler is not None and trainer.is_global_zero:
            print(profiler.summary())

if __name__ == "__main__":
    # Usage:
    # python -m scripts.train_model                              (OAT training)
    # python -m scripts.train_model model=baseline logging=baseline  (baseline evaluation)
    # python -m scripts.train_model +saves_fixtures=true ++dataloader.batch_size=1
    main()
