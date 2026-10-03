bash submit_train_h100.sh 8 14 \
    domain=protein \
    paradigm=multiframefm \
    datamodule.batch_size=4 \
    datamodule.num_workers=8 \
    model.c_s=768 \
    model.c_cond=768 \
    model.c_frame=256 \
    model.c_framepair=64 \
    model.z_broadcast=true \
    model.rigid_transformer_num_blocks=1 \
    model.rigid_transformer_rigid_updates=true \
    model.use_embedder_sc_rigid_transformer=true \
    model.use_ipa_gating=true \
    model.use_qk_norm=true \
    model.use_amp=true \
    model.rot_preconditioning=true \
    model.num_blocks=12 \
    lmodule.use_ema=true \
    corrupter.use_stochastic_centering=true \
    corrupter.center_on_motif=true \
    corrupter.sig_perturb=4 \
    dataset.config=configs/train/data/afdb_512_clusters_crop256.yaml \
    experiment.lightning.devices=8 \
    experiment.lightning.strategy=ddp_find_unused_parameters_true \
    experiment.checkpointer.train_time_interval=null \
    experiment.checkpointer.every_n_train_steps=20000 \
    hydra.run.dir="./outputs/pretrain/phase1_0"

sleep 1s