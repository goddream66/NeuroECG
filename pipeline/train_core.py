#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level ECGFounder + statement71 training pipeline orchestration."""

from pipeline.train_config import *

from data.cache import get_or_build_record_cache
from data.datasets import ICAREDataset, build_indices, build_wvnum_entry_samples
from data.splits import load_or_create_split, split_from_resume_if_compatible
from data.train_metadata import filter_pid_meta_to_ecg_available
from evaluation.metrics import evaluate_segment_metrics
from evaluation.reports import print_training_key_summary, summarize, write_csv
from feature.deep import add_aux_projected_feature_blocks, add_deep_pca_feature_blocks, extract_deep_patient_block_for_split, merge_deep_and_statement_blocks
from feature.statement71 import extract_patient_statement71_block_lowmem, load_fastai_xresnet_model, load_statement_classes
from model.convnext1d import ConvNeXt1DFeatureExtractor, make_convnext1d_optimizer, set_convnext1d_epoch_lr, set_convnext1d_training_mode
from model.ecgfounder import ECGFounderFeatureExtractor, make_ecgfounder_optimizer, set_ecgfounder_gradual_unfreezing
from model.ecg_jepa_adapter import ECGJEPAFeatureExtractor, make_ecg_jepa_optimizer, set_ecg_jepa_gradual_unfreezing
from model.ecgfm_adapter import ECGFMFeatureExtractor, make_ecgfm_optimizer, set_ecgfm_gradual_unfreezing
from model.fusion import train_eval_clinical_residual_deep_fusion, train_eval_deep_static_gated_fusion
from model.heads import BinaryClassificationHead, IdentityProjector
from model.seresnet import SEResNetFeatureExtractor, make_seresnet_optimizer, set_seresnet_epoch_lr, set_seresnet_training_mode
from model.stmem_adapter import STMEMFeatureExtractor, make_stmem_optimizer, set_stmem_gradual_unfreezing
from pipeline.args import parse_stage1_args
from pipeline.experiments import train_eval_design_specs
from utils.checkpoint import (
    EarlyStopping,
    load_resume_checkpoint,
    save_training_resume_checkpoint,
)
from utils.runtime import describe_selected_gpu, release_memory, set_seed


def group_samples_by_patient(samples):
    grouped = {}
    for sample in samples:
        grouped.setdefault(str(sample["pid"]), []).append(sample)
    return grouped


def sample_segments_per_patient_for_epoch(grouped_samples, max_segments_per_patient, seed):
    max_segments_per_patient = int(max_segments_per_patient)
    if max_segments_per_patient <= 0:
        return [sample for pid in sorted(grouped_samples) for sample in grouped_samples[pid]]

    rng = np.random.default_rng(int(seed))
    epoch_samples = []
    for pid in sorted(grouped_samples):
        rows = grouped_samples[pid]
        if len(rows) <= max_segments_per_patient:
            selected_indices = np.arange(len(rows), dtype=np.int64)
        else:
            selected_indices = rng.choice(len(rows), size=max_segments_per_patient, replace=False)
            selected_indices.sort()
        epoch_samples.extend(rows[int(i)] for i in selected_indices)
    rng.shuffle(epoch_samples)
    return epoch_samples


def run_training_pipeline() -> None:
    args = parse_stage1_args()
    # 72h segment model training defaults requested for this experiment.
    args.segment_train_batch_size = int(STAGE1_SEGMENT_TRAIN_BATCH_SIZE)
    args.early_stop_patience = 6
    args.early_stop_min_epochs = 0
    set_seed(BASE_SEED)
    start_total = time.time()
    device = describe_selected_gpu("PTBXL71 semantic training pipeline")
    amp_enabled = device.type == "cuda"
    num_workers = DATALOADER_WORKERS

    backbone_key = str(STAGE1_BACKBONE).strip().lower()
    if backbone_key == "ecg_fm":
        backbone_key = "ecgfm"
    if backbone_key not in {"ecgfounder", "seresnet", "convnext1d", "ecg_jepa", "ecgfm", "stmem"}:
        raise ValueError(
            f"Unsupported STAGE1_BACKBONE={STAGE1_BACKBONE!r}; "
            "expected 'ecgfounder', 'seresnet', 'convnext1d', 'ecg_jepa', 'ecgfm', or 'stmem'."
        )
    if backbone_key == "seresnet":
        stage1_backbone_label = "SE-ResNet"
    elif backbone_key == "convnext1d":
        stage1_backbone_label = "ConvNeXt1D"
    elif backbone_key == "ecg_jepa":
        stage1_backbone_label = "ECG-JEPA"
    elif backbone_key == "ecgfm":
        stage1_backbone_label = "ECG-FM"
    elif backbone_key == "stmem":
        stage1_backbone_label = "ST-MEM"
    else:
        stage1_backbone_label = "ECGFounder"

    print("\n=========================================================")
    print(f" {stage1_backbone_label} Single-lead I-CARE Segment Classifier")
    print(" segment-level outcome training; no MIL pooling, no PTBXL auxiliary branch")
    print("=========================================================")
    print(f"[Device] {device}")
    print(f"[Script Dir] {SCRIPT_DIR}")
    print(f"[Shared Record Cache] {RECORD_CACHE}")
    print(f"[Split Resume Source] {RESUME_CHECKPOINT_PATH}")
    print(
        f"[Stage1 Training Mode] {stage1_backbone_label} 1-lead BCE | "
        f"ptbxl_aux=0 | mil_pooling=0 | early_stop=val_loss_min_patience6 | "
        f"epochs={STAGE1_EPOCHS} | batch_size={STAGE1_SEGMENT_TRAIN_BATCH_SIZE}"
    )
    if STAGE2_ONLY_LOAD_STAGE1_CKPT:
        print(f"[Stage2 Only] enabled: skip Stage 1 training and load checkpoint: {TRAIN_BEST_CKPT}")
        print(f"[Stage2 Only] outputs will be written with tag: {TRAIN_OUTPUT_TAG}")
    print(
        "[Stage1 Data] HRV computation disabled in train/val/test DataLoaders; "
        "HRV is recomputed only in Stage 2 feature extraction."
    )
    print(f"[Observation Window] using ECG records with segment_end_hour <= {MAX_OBSERVATION_HOUR:g}h | buckets={SELECTED_TIME_BUCKETS}")
    print(
        f"[Segment Stage1] segment_train_batch_size={int(args.segment_train_batch_size)} | "
        f"label_mode=parent_patient_outcome_as_segment_label | no_mil_pooling=True"
    )

    all_records, pid_meta_full = get_or_build_record_cache(
        META_DIR,
        PROCESSED_DIR,
        RECORD_CACHE,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        selected_time_buckets=SELECTED_TIME_BUCKETS,
        scan_processes=SCAN_PROCESSES,
    )
    pid_meta = filter_pid_meta_to_ecg_available(pid_meta_full, all_records)

    resume_state = load_resume_checkpoint(TRAIN_RESUME_CKPT, device)
    compatible_split = split_from_resume_if_compatible(resume_state, pid_meta)
    if compatible_split is not None:
        train_pids, val_pids, test_pids = compatible_split
        split_source = "train_resume_checkpoint"
        print(f"[Resume] Found compatible training resume at {TRAIN_RESUME_CKPT}")
    else:
        resume_state = None
        train_pids, val_pids, test_pids, split_source = load_or_create_split(pid_meta, device)
    print(
        f"[Split] source={split_source} | patients train/val/test="
        f"{len(train_pids)}/{len(val_pids)}/{len(test_pids)}"
    )

    train_samples = build_indices(all_records, train_pids)
    val_samples = build_indices(all_records, val_pids)
    test_samples = build_indices(all_records, test_pids)
    del all_records, pid_meta_full
    release_memory("record cache after split samples were built")
    print(
        f"[Split] segments train/val/test="
        f"{len(train_samples)}/{len(val_samples)}/{len(test_samples)}"
    )

    # Stage 1 only needs ECG waveform and labels. Disable HRV here to avoid
    # expensive CPU peak detection for every segment during ECGFounder training.
    train_samples_by_pid = group_samples_by_patient(train_samples)
    train_patient_count = len(train_samples_by_pid)
    val_ds = ICAREDataset(val_samples, is_train=False, clip_value=INPUT_CLIP_VALUE, compute_hrv=False)
    test_ds = ICAREDataset(test_samples, is_train=False, clip_value=INPUT_CLIP_VALUE, compute_hrv=False)

    segment_train_batch_size = int(args.segment_train_batch_size)
    val_loader = DataLoader(
        val_ds,
        batch_size=segment_train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DATALOADER_PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=segment_train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DATALOADER_PIN_MEMORY,
    )
    print(
        f"[Stage1 Segment Dataset] train_full/val/test segments="
        f"{len(train_samples)}/{len(val_ds)}/{len(test_ds)} | "
        f"train_patients={train_patient_count} | "
        f"max_train_segments_per_patient_per_epoch={STAGE1_MAX_SEGMENTS_PER_PATIENT_PER_EPOCH} | "
        f"segment_train_batch_size={segment_train_batch_size} | "
        f"label_mode=parent_patient_outcome_as_segment_label | no_mil_pooling=True"
    )

    print("[Stage1] PTBXL auxiliary branch removed. PTBXL71 is loaded only in Stage 2 feature extraction.")

    if backbone_key == "seresnet":
        extractor = SEResNetFeatureExtractor().to(device)
        set_stage1_training_mode = set_seresnet_training_mode
        make_stage1_optimizer = make_seresnet_optimizer
        stage1_schedule_summary = (
            f"full_train_from_scratch_all_epochs; "
            f"epochs_1_to_{SERESNET_LR_LOW_EPOCHS}=lr{SERESNET_LR_LOW:g}; "
            f"epochs_{SERESNET_LR_LOW_EPOCHS + 1}_to_{SERESNET_LR_MID_EPOCHS}=lr{SERESNET_LR_MID:g}; "
            f"epochs_{SERESNET_LR_MID_EPOCHS + 1}_to_{STAGE1_EPOCHS}=lr{SERESNET_LR_HIGH:g}"
        )
        stage1_lr_summary = (
            f"low={SERESNET_LR_LOW:g}, mid={SERESNET_LR_MID:g}, high={SERESNET_LR_HIGH:g}, "
            f"dropout={SERESNET_DROPOUT:g}"
        )
    elif backbone_key == "convnext1d":
        extractor = ConvNeXt1DFeatureExtractor().to(device)
        set_stage1_training_mode = set_convnext1d_training_mode
        make_stage1_optimizer = make_convnext1d_optimizer
        stage1_schedule_summary = (
            f"full_train_from_scratch_all_epochs; "
            f"depths={CONVNEXT1D_DEPTHS}; dims={CONVNEXT1D_DIMS}; "
            f"epochs_1_to_{CONVNEXT1D_LR_LOW_EPOCHS}=lr{CONVNEXT1D_LR_LOW:g}; "
            f"epochs_{CONVNEXT1D_LR_LOW_EPOCHS + 1}_to_{CONVNEXT1D_LR_MID_EPOCHS}=lr{CONVNEXT1D_LR_MID:g}; "
            f"epochs_{CONVNEXT1D_LR_MID_EPOCHS + 1}_to_{STAGE1_EPOCHS}=lr{CONVNEXT1D_LR_HIGH:g}"
        )
        stage1_lr_summary = (
            f"low={CONVNEXT1D_LR_LOW:g}, mid={CONVNEXT1D_LR_MID:g}, high={CONVNEXT1D_LR_HIGH:g}, "
            f"drop_path={CONVNEXT1D_DROP_PATH_RATE:g}, dropout={CONVNEXT1D_DROPOUT:g}"
        )
    elif backbone_key == "ecg_jepa":
        extractor = ECGJEPAFeatureExtractor().to(device)
        set_stage1_training_mode = set_ecg_jepa_gradual_unfreezing
        make_stage1_optimizer = make_ecg_jepa_optimizer
        stage1_schedule_summary = (
            f"head_only_epochs={ECGFOUNDER_FREEZE_HEAD_ONLY_EPOCHS}, "
            f"last_two_blocks_from_epoch={ECGFOUNDER_UNFREEZE_LAST_TWO_STAGES_EPOCH + 1}, "
            f"last_four_blocks_from_epoch={ECGFOUNDER_UNFREEZE_LAST_FOUR_STAGES_EPOCH + 1}, "
            f"full_encoder_from_epoch={ECGFOUNDER_UNFREEZE_FULL_BACKBONE_EPOCH + 1}"
        )
        stage1_lr_summary = (
            f"head={ECGFOUNDER_HEAD_LR:g}, last_two={ECGFOUNDER_LAST_TWO_STAGES_LR:g}, "
            f"last_four_extra={ECGFOUNDER_LAST_FOUR_STAGES_LR:g}, "
            f"full_encoder_rest={ECGFOUNDER_FULL_BACKBONE_LR:g}; "
            f"input_resample={WINDOW_SIZE}->{ECG_JEPA_TARGET_LEN}, leads={tuple(ECG_JEPA_LEADS)}"
        )
    elif backbone_key == "ecgfm":
        extractor = ECGFMFeatureExtractor().to(device)
        set_stage1_training_mode = set_ecgfm_gradual_unfreezing
        make_stage1_optimizer = make_ecgfm_optimizer
        stage1_schedule_summary = (
            f"head_only_epochs={ECGFOUNDER_FREEZE_HEAD_ONLY_EPOCHS}, "
            f"last_two_blocks_from_epoch={ECGFOUNDER_UNFREEZE_LAST_TWO_STAGES_EPOCH + 1}, "
            f"last_four_blocks_from_epoch={ECGFOUNDER_UNFREEZE_LAST_FOUR_STAGES_EPOCH + 1}, "
            f"full_backbone_from_epoch={ECGFOUNDER_UNFREEZE_FULL_BACKBONE_EPOCH + 1}"
        )
        stage1_lr_summary = (
            f"head={ECGFOUNDER_HEAD_LR:g}, last_two={ECGFOUNDER_LAST_TWO_STAGES_LR:g}, "
            f"last_four_extra={ECGFOUNDER_LAST_FOUR_STAGES_LR:g}, "
            f"full_backbone_rest={ECGFOUNDER_FULL_BACKBONE_LR:g}; "
            f"single_lead_to_12lead_index={ECGFM_SINGLE_LEAD_INDEX}, target_len={ECGFM_TARGET_LEN}"
        )
    elif backbone_key == "stmem":
        extractor = STMEMFeatureExtractor().to(device)
        set_stage1_training_mode = set_stmem_gradual_unfreezing
        make_stage1_optimizer = make_stmem_optimizer
        stage1_schedule_summary = (
            f"head_only_epochs={ECGFOUNDER_FREEZE_HEAD_ONLY_EPOCHS}, "
            f"last_two_blocks_from_epoch={ECGFOUNDER_UNFREEZE_LAST_TWO_STAGES_EPOCH + 1}, "
            f"last_four_blocks_from_epoch={ECGFOUNDER_UNFREEZE_LAST_FOUR_STAGES_EPOCH + 1}, "
            f"full_encoder_from_epoch={ECGFOUNDER_UNFREEZE_FULL_BACKBONE_EPOCH + 1}"
        )
        stage1_lr_summary = (
            f"head={ECGFOUNDER_HEAD_LR:g}, last_two={ECGFOUNDER_LAST_TWO_STAGES_LR:g}, "
            f"last_four_extra={ECGFOUNDER_LAST_FOUR_STAGES_LR:g}, "
            f"full_encoder_rest={ECGFOUNDER_FULL_BACKBONE_LR:g}; "
            f"single_lead_to_12lead_index={ST_MEM_SINGLE_LEAD_INDEX}, "
            f"target_len={ST_MEM_TARGET_LEN}, patch_size={ST_MEM_PATCH_SIZE}"
        )
    else:
        extractor = ECGFounderFeatureExtractor(
            ecgfounder_root=ECGFOUNDER_ROOT,
            checkpoint_path=ECGFOUNDER_1LEAD_CKPT,
        ).to(device)
        set_stage1_training_mode = set_ecgfounder_gradual_unfreezing
        make_stage1_optimizer = make_ecgfounder_optimizer
        stage1_schedule_summary = (
            f"head_only_epochs={ECGFOUNDER_FREEZE_HEAD_ONLY_EPOCHS}, "
            f"last_two_stages_from_epoch={ECGFOUNDER_UNFREEZE_LAST_TWO_STAGES_EPOCH + 1}, "
            f"last_four_stages_from_epoch={ECGFOUNDER_UNFREEZE_LAST_FOUR_STAGES_EPOCH + 1}, "
            f"full_backbone_from_epoch={ECGFOUNDER_UNFREEZE_FULL_BACKBONE_EPOCH + 1}"
        )
        stage1_lr_summary = (
            f"head={ECGFOUNDER_HEAD_LR:g}, last_two={ECGFOUNDER_LAST_TWO_STAGES_LR:g}, "
            f"last_four_extra={ECGFOUNDER_LAST_FOUR_STAGES_LR:g}, "
            f"full_backbone_rest={ECGFOUNDER_FULL_BACKBONE_LR:g}"
        )

    stage1_feature_dim = int(getattr(extractor, "feature_dim", ECGFOUNDER_FEATURE_DIM))
    projected_feature_dim = int(stage1_feature_dim)
    projector = IdentityProjector().to(device)
    classifier_head = BinaryClassificationHead(projected_feature_dim).to(device)
    print(
        f"[Stage1 Projector] disabled: classifier receives raw {stage1_backbone_label} deep features "
        f"({stage1_feature_dim} dims)."
    )

    optimizer = make_stage1_optimizer(
        extractor=extractor,
        classifier_head=classifier_head,
        weight_decay=args.weight_decay,
    )
    print(
        f"[{stage1_backbone_label} Train] schedule={stage1_schedule_summary} | "
        f"lrs={stage1_lr_summary}"
    )
    outcome_criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=amp_enabled)
    early_stop = EarlyStopping(
        patience=int(args.early_stop_patience),
        min_delta=0.0,
        mode="min",
        path=TRAIN_BEST_CKPT,
    )

    total_epochs = int(STAGE1_EPOCHS)
    start_epoch = 0

    if resume_state is not None and not STAGE2_ONLY_LOAD_STAGE1_CKPT:
        extractor.load_state_dict(resume_state["extractor_state_dict"])
        projector.load_state_dict(resume_state.get("projector_state_dict", {}), strict=False)
        classifier_head.load_state_dict(resume_state["classifier_head_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scaler_state = resume_state.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        early_stop_state = resume_state.get("early_stopping_state")
        if early_stop_state:
            early_stop.load_state_dict(early_stop_state)
        start_epoch = int(resume_state["epoch"]) + 1
        print(f"[Resume] Restored training state. Next epoch={start_epoch + 1}/{total_epochs}")

    if STAGE2_ONLY_LOAD_STAGE1_CKPT:
        if not os.path.exists(TRAIN_BEST_CKPT):
            raise FileNotFoundError(
                f"Stage2-only mode requires Stage-1 best checkpoint, but it was not found: {TRAIN_BEST_CKPT}"
            )
        start_epoch = total_epochs
        print("[Stage2 Only] Stage 1 training is skipped. Existing Stage-1 weights will be loaded before Stage 2A.")

    print(f"\n[Stage 1] Plain segment-level BCE training (Total Epochs: {total_epochs})...")
    if start_epoch >= total_epochs:
        print("[Stage 1] Resume checkpoint indicates segment training is already complete. Skipping.")
    else:
        try:
            for epoch in range(start_epoch, total_epochs):
                finetune_phase = set_stage1_training_mode(extractor, classifier_head, epoch)
                if backbone_key == "seresnet":
                    finetune_phase = f"{finetune_phase}+{set_seresnet_epoch_lr(optimizer, epoch)}"
                elif backbone_key == "convnext1d":
                    finetune_phase = f"{finetune_phase}+{set_convnext1d_epoch_lr(optimizer, epoch)}"
                projector.eval()
                epoch_start = time.time()
                train_epoch_samples = sample_segments_per_patient_for_epoch(
                    train_samples_by_pid,
                    STAGE1_MAX_SEGMENTS_PER_PATIENT_PER_EPOCH,
                    seed=BASE_SEED + epoch,
                )
                train_epoch_ds = ICAREDataset(
                    train_epoch_samples,
                    is_train=True,
                    clip_value=INPUT_CLIP_VALUE,
                    compute_hrv=False,
                )
                train_epoch_loader = DataLoader(
                    train_epoch_ds,
                    batch_size=segment_train_batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=DATALOADER_PIN_MEMORY,
                )

                print(
                    f"  [Epoch {epoch + 1}/{total_epochs}] sampled_segments={len(train_epoch_ds)} | "
                    f"train_patients={train_patient_count} | "
                    f"max_per_patient={STAGE1_MAX_SEGMENTS_PER_PATIENT_PER_EPOCH} | "
                    f"batch_size={segment_train_batch_size} | "
                    f"label_mode=patient_label_as_segment_label | no_mil_pooling=True | "
                    f"finetune_phase={finetune_phase}"
                )

                for batch_idx, (ecg, channel_mask, labels, _, _, _, _, _, _) in enumerate(train_epoch_loader, start=1):
                    ecg = ecg.to(device).float()
                    channel_mask = channel_mask.to(device).float()
                    labels = labels.to(device).float()
                    optimizer.zero_grad(set_to_none=True)

                    with autocast(enabled=amp_enabled):
                        segment_embeddings = projector(extractor(ecg, channel_mask=channel_mask))
                        segment_logits = classifier_head(segment_embeddings)
                        outcome_loss = outcome_criterion(segment_logits, labels)
                        total_loss = outcome_loss

                    scaler.scale(total_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(extractor.parameters())
                        + list(classifier_head.parameters()),
                        1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    if batch_idx % 50 == 0:
                        print(
                            f"  Epoch {epoch + 1}/{total_epochs} | Batch {batch_idx}/{len(train_epoch_loader)} | "
                            f"total={float(total_loss.item()):.4f} | outcome={float(outcome_loss.item()):.4f} | "
                            f"plain_bce=on"
                        )
                del train_epoch_loader, train_epoch_ds, train_epoch_samples

                avg_val_loss, val_auroc, val_auprc = evaluate_segment_metrics(
                    extractor,
                    projector,
                    classifier_head,
                    val_loader,
                    outcome_criterion,
                    device,
                    amp_enabled,
                )
                monitor_score = np.inf if np.isnan(avg_val_loss) else float(avg_val_loss)
                print(
                    f"  [Epoch {epoch + 1} Completed] Val Loss: {avg_val_loss:.4f} | "
                    f"Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f} | "
                    f"Monitor: val_loss={monitor_score:.4f} | Time: {(time.time() - epoch_start) / 60:.1f}m"
                )

                early_stop(
                    monitor_score,
                    {
                        "extractor_state_dict": extractor.state_dict(),
                        "projector_state_dict": projector.state_dict(),
                        "classifier_head_state_dict": classifier_head.state_dict(),
                        "best_epoch": epoch,
                        "val_loss": float(avg_val_loss),
                        "val_auroc": float(val_auroc),
                        "val_auprc": float(val_auprc),
                        "early_stop_monitor": "val_loss",
                        "early_stop_mode": "min",
                        "segment_train_batch_size": int(segment_train_batch_size),
                        "total_epochs": int(total_epochs),
                    },
                )
                save_training_resume_checkpoint(
                    TRAIN_RESUME_CKPT,
                    epoch,
                    extractor,
                    projector,
                    classifier_head,
                    optimizer,
                    scaler,
                    early_stop,
                    train_pids,
                    val_pids,
                    test_pids,
                )
                print(f"  [Checkpoint] Saved training resume state to {TRAIN_RESUME_CKPT}")

                if early_stop.early_stop and (epoch + 1) < int(args.early_stop_min_epochs):
                    print(
                        f"[EarlyStop Guard] patience reached at epoch {epoch + 1}, "
                        f"but min_epochs={args.early_stop_min_epochs} not reached. Continue training."
                    )
                    early_stop.early_stop = False
                    early_stop.counter = 0

                if early_stop.early_stop:
                    print("[Stop] Early stopping triggered.")
                    break
        except KeyboardInterrupt:
            interrupted_epoch = locals().get("epoch", start_epoch)
            save_training_resume_checkpoint(
                TRAIN_RESUME_CKPT,
                interrupted_epoch,
                extractor,
                projector,
                classifier_head,
                optimizer,
                scaler,
                early_stop,
                train_pids,
                val_pids,
                test_pids,
            )
            print(f"\n[Interrupt] Saved training resume checkpoint to {TRAIN_RESUME_CKPT}")
            raise

    best_state = torch.load(TRAIN_BEST_CKPT, map_location=device)
    extractor.load_state_dict(best_state["extractor_state_dict"])
    projector.load_state_dict(best_state.get("projector_state_dict", {}), strict=False)
    classifier_head.load_state_dict(best_state["classifier_head_state_dict"])
    extractor.eval()
    projector.eval()
    classifier_head.eval()

    val_direct_loss = val_direct_auroc = val_direct_auprc = np.nan
    test_direct_loss = test_direct_auroc = test_direct_auprc = np.nan
    if STAGE2_ONLY_LOAD_STAGE1_CKPT and SKIP_STAGE1_DIRECT_EVAL_IN_STAGE2_ONLY:
        print("\n[Stage 1 Eval] Skipped in Stage2-only mode to start feature extraction directly.")
    else:
        print("\n[Stage 1 Eval] Direct segment-level outcome metrics")
        val_direct_loss, val_direct_auroc, val_direct_auprc = evaluate_segment_metrics(
            extractor,
            projector,
            classifier_head,
            val_loader,
            outcome_criterion,
            device,
            amp_enabled,
        )
        test_direct_loss, test_direct_auroc, test_direct_auprc = evaluate_segment_metrics(
            extractor,
            projector,
            classifier_head,
            test_loader,
            outcome_criterion,
            device,
            amp_enabled,
        )
        print(
            f"[Stage 1 Direct Segment] val_loss={val_direct_loss:.4f} | val_auroc={val_direct_auroc:.4f} | "
            f"val_auprc={val_direct_auprc:.4f} | test_loss={test_direct_loss:.4f} | "
            f"test_auroc={test_direct_auroc:.4f} | test_auprc={test_direct_auprc:.4f}"
        )

    del val_loader, test_loader, val_ds, test_ds
    release_memory("Stage 1 DataLoaders and datasets")

    print("\n[Stage 2A] Extracting trained deep patient features")
    deep_blocks = {
        "train": extract_deep_patient_block_for_split(
            extractor, projector, train_samples, "train", device, amp_enabled, num_workers, BATCH_SIZE, pid_meta
        ),
        "val": extract_deep_patient_block_for_split(
            extractor, projector, val_samples, "val", device, amp_enabled, num_workers, BATCH_SIZE, pid_meta
        ),
        "test": extract_deep_patient_block_for_split(
            extractor, projector, test_samples, "test", device, amp_enabled, num_workers, BATCH_SIZE, pid_meta
        ),
    }
    del (
        extractor,
        projector,
        classifier_head,
        optimizer,
        scaler,
        early_stop,
        best_state,
    )
    release_memory(f"Stage 2A segment matrices and {stage1_backbone_label} objects")

    print("\n[Stage 2B] Extracting PTBXL71 semantic patient features for final fusion")
    statement_classes = load_statement_classes()
    semantic_model = load_fastai_xresnet_model(device)
    train_entry_samples = build_wvnum_entry_samples(train_samples, "train", MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT)
    train_statement_block = extract_patient_statement71_block_lowmem(
        semantic_model, train_entry_samples, "train", device, int(args.semantic_batch_size), NUM_WORKERS, pid_meta
    )
    del train_entry_samples
    release_memory("Stage 2B train entry samples")
    val_entry_samples = build_wvnum_entry_samples(val_samples, "val", MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT)
    val_statement_block = extract_patient_statement71_block_lowmem(
        semantic_model, val_entry_samples, "val", device, int(args.semantic_batch_size), NUM_WORKERS, pid_meta
    )
    del val_entry_samples
    release_memory("Stage 2B val entry samples")
    test_entry_samples = build_wvnum_entry_samples(test_samples, "test", MAX_ENTRIES_PER_PATIENT_FOR_STATEMENT)
    test_statement_block = extract_patient_statement71_block_lowmem(
        semantic_model, test_entry_samples, "test", device, int(args.semantic_batch_size), NUM_WORKERS, pid_meta
    )
    del test_entry_samples
    release_memory("Stage 2B test entry samples")
    blocks = {
        "train": merge_deep_and_statement_blocks(deep_blocks["train"], train_statement_block, "train"),
        "val": merge_deep_and_statement_blocks(deep_blocks["val"], val_statement_block, "val"),
        "test": merge_deep_and_statement_blocks(deep_blocks["test"], test_statement_block, "test"),
    }
    deep_pca_info = add_deep_pca_feature_blocks(
        blocks,
        n_components=DEEP_PCA_DIM,
    )
    aux_projector_info = add_aux_projected_feature_blocks(
        blocks,
        out_dim=AUX_FEATURE_PROJECTOR_DIM,
        seed=AUX_FEATURE_PROJECTOR_SEED,
    )
    del (
        train_statement_block,
        val_statement_block,
        test_statement_block,
        deep_blocks,
        semantic_model,
        statement_classes,
        train_samples,
        val_samples,
        test_samples,
    )
    release_memory("Stage 2B grouped statement/HRV objects")

    print("\n[Stage 3] CatBoost fusion evaluation")
    base_design_specs = [
        ("deep_only", f"deep_patient_{PATIENT_POOLING_TAG}"),
        ("deep_pca64_only", f"deep_pca64_patient_{PATIENT_POOLING_TAG}"),
        ("deep_pca64_static", f"static_plus_deep_pca64_{PATIENT_POOLING_TAG}"),
        ("deep_pca64_hrv_static", f"static_hrv_{PATIENT_POOLING_TAG}_deep_pca64"),
        ("deep_pca64_statement71_static", f"static_deep_pca64_{PATIENT_POOLING_TAG}_statement71"),
        ("deep_pca64_statement71_hrv_static", f"static_hrv_{PATIENT_POOLING_TAG}_deep_pca64_statement71"),
    ]
    rows = train_eval_design_specs(base_design_specs, blocks, pid_meta, "Fusion Ablation")
    evaluated_design_specs = list(base_design_specs)
    if USE_DEEP_STATIC_GATED_FUSION:
        gated_specs = [
            ("gated_deep_static", f"deep_static_projected{FUSION_ALIGN_DIM}_softmax_gated"),
            (
                "gated_deep_static_hrv_statement71",
                f"deep_static_projected{FUSION_ALIGN_DIM}_softmax_gated_plus_hrv_statement71",
            ),
            (
                "gated_deep_statement71_static_hrv",
                f"deep_statement71_projected{FUSION_ALIGN_DIM}_softmax_gated_plus_static_hrv",
            ),
            (
                "gated_deep_pca64_static",
                f"deep_pca64_static_projected{FUSION_ALIGN_DIM}_softmax_gated",
            ),
            (
                "gated_deep_pca64_static_hrv_statement71",
                f"deep_pca64_static_projected{FUSION_ALIGN_DIM}_softmax_gated_plus_hrv_statement71",
            ),
            (
                "fixedgate_deep_static",
                f"deep_static_projected{FUSION_ALIGN_DIM}_fixedgate_d{FIXED_GATE_DEEP_WEIGHT:.3f}_s{FIXED_GATE_STATIC_WEIGHT:.3f}",
            ),
            (
                "fixedgate_deep_static_hrv_statement71",
                f"deep_static_projected{FUSION_ALIGN_DIM}_fixedgate_d{FIXED_GATE_DEEP_WEIGHT:.3f}_s{FIXED_GATE_STATIC_WEIGHT:.3f}_plus_hrv_statement71",
            ),
        ]
        evaluated_design_specs.extend(gated_specs)
        rows.extend(train_eval_deep_static_gated_fusion(blocks, pid_meta))
    if USE_CLINICAL_RESIDUAL_DEEP_FUSION:
        residual_specs = [
            (
                "clinical_residual_deep_pca64",
                "clinical_static_hrv_statement71_plus_patient_residual_deep_pca64",
            ),
            (
                "clinical_residual_deep1024",
                "clinical_static_hrv_statement71_plus_patient_residual_deep1024",
            ),
        ]
        evaluated_design_specs.extend(residual_specs)
        rows.extend(train_eval_clinical_residual_deep_fusion(blocks, pid_meta))

    write_csv(TRAIN_RAW_CSV, rows)
    summary_rows = summarize(rows)
    write_csv(TRAIN_SUMMARY_CSV, summary_rows)
    print_training_key_summary(summary_rows)

    config = {
        "script_dir": SCRIPT_DIR,
        "record_cache": RECORD_CACHE,
        "split_source": split_source,
        "train_best_checkpoint": TRAIN_BEST_CKPT,
        "train_resume_checkpoint": TRAIN_RESUME_CKPT,
        "train_output_tag": TRAIN_OUTPUT_TAG,
        "train_stage1_checkpoint_tag": TRAIN_STAGE1_CKPT_TAG,
        "stage2_only_load_stage1_ckpt": bool(STAGE2_ONLY_LOAD_STAGE1_CKPT),
        "skip_stage1_direct_eval_in_stage2_only": bool(SKIP_STAGE1_DIRECT_EVAL_IN_STAGE2_ONLY),
        "ptbxl_repo_root": PTBXL_REPO_ROOT,
        "ptbxl_pth_path": PTBXL_PTH_PATH,
        "ptbxl_mlb_path": PTBXL_MLB_PATH,
        "stage1_training_mode": "stage2_only_load_existing_stage1_checkpoint" if STAGE2_ONLY_LOAD_STAGE1_CKPT else f"segment_level_patient_label_{backbone_key}_no_mil",
        "stage1_pooling": f"none_for_training; patient_{PATIENT_POOLING_TAG}_only_for_stage2_feature_export",
        "stage1_sampling": {
            "mode": "patient_balanced_random_segments_per_epoch",
            "max_segments_per_patient_per_epoch": int(STAGE1_MAX_SEGMENTS_PER_PATIENT_PER_EPOCH),
            "seed": "BASE_SEED + epoch",
        },
        "segment_train_batch_size": int(args.segment_train_batch_size),
        "stage1_backbone": STAGE1_BACKBONE,
        "stage1_feature_dim": int(stage1_feature_dim),
        "stage1_classifier_input_dim": int(projected_feature_dim),
        "convnext1d": {
            "enabled": backbone_key == "convnext1d",
            "dims": list(CONVNEXT1D_DIMS),
            "depths": list(CONVNEXT1D_DEPTHS),
            "kernel_size": int(CONVNEXT1D_KERNEL_SIZE),
            "drop_path_rate": float(CONVNEXT1D_DROP_PATH_RATE),
            "dropout": float(CONVNEXT1D_DROPOUT),
            "lr_low": float(CONVNEXT1D_LR_LOW),
            "lr_mid": float(CONVNEXT1D_LR_MID),
            "lr_high": float(CONVNEXT1D_LR_HIGH),
        },
        "hrv_feature_extraction": {
            "source": "segment_level_ecg_windows",
            "segment_seconds": float(WINDOW_SIZE) / float(TARGET_FS),
            "patient_pooling": f"q{TEMPORAL_AGG_QUANTILE:.2f}",
        },
        "deep_pca": deep_pca_info,
        "aux_feature_projector": aux_projector_info,
        "deep_static_gated_fusion": {
            "enabled": bool(USE_DEEP_STATIC_GATED_FUSION),
            "variants": [
                {
                    "name": "gated_deep_static",
                    "gated_modalities": ["deep", "static"],
                    "extra_concat": [],
                },
                {
                    "name": "gated_deep_static_hrv_statement71",
                    "gated_modalities": ["deep", "static"],
                    "extra_concat": ["hrv", SEMANTIC_FEATURE_NAME],
                },
                {
                    "name": "gated_deep_statement71_static_hrv",
                    "gated_modalities": ["deep", SEMANTIC_FEATURE_NAME],
                    "extra_concat": ["static", "hrv"],
                },
                {
                    "name": "gated_deep_pca64_static",
                    "gated_modalities": [f"deep_pca{DEEP_PCA_DIM}", "static"],
                    "extra_concat": [],
                },
                {
                    "name": "gated_deep_pca64_static_hrv_statement71",
                    "gated_modalities": [f"deep_pca{DEEP_PCA_DIM}", "static"],
                    "extra_concat": ["hrv", SEMANTIC_FEATURE_NAME],
                },
                {
                    "name": "fixedgate_deep_static",
                    "gated_modalities": ["deep", "static"],
                    "extra_concat": [],
                    "fixed_gate": {
                        "deep": float(FIXED_GATE_DEEP_WEIGHT),
                        "static": float(FIXED_GATE_STATIC_WEIGHT),
                    },
                },
                {
                    "name": "fixedgate_deep_static_hrv_statement71",
                    "gated_modalities": ["deep", "static"],
                    "extra_concat": ["hrv", SEMANTIC_FEATURE_NAME],
                    "fixed_gate": {
                        "deep": float(FIXED_GATE_DEEP_WEIGHT),
                        "static": float(FIXED_GATE_STATIC_WEIGHT),
                    },
                },
            ],
            "align_dim": int(FUSION_ALIGN_DIM),
            "hidden_dim": int(FUSION_HIDDEN_DIM),
            "dropout": float(FUSION_DROPOUT),
            "batch_size": int(FUSION_BATCH_SIZE),
            "epochs": int(FUSION_EPOCHS),
            "patience": int(FUSION_PATIENCE),
            "lr": float(FUSION_LR),
            "weight_decay": float(FUSION_WEIGHT_DECAY),
            "cpc_loss_weight": float(FUSION_CPC_LOSS_WEIGHT),
            "fusion": "project_selected_modalities_to_shared_dim_then_learn_softmax_gate_then_optionally_concat_extra_modalities",
        },
        "clinical_residual_deep_fusion": {
            "enabled": bool(USE_CLINICAL_RESIDUAL_DEEP_FUSION),
            "variants": [
                {
                    "name": "clinical_residual_deep_pca64",
                    "clinical_base": ["static", "hrv", SEMANTIC_FEATURE_NAME],
                    "deep_residual": f"deep_pca{DEEP_PCA_DIM}",
                },
                {
                    "name": "clinical_residual_deep1024",
                    "clinical_base": ["static", "hrv", SEMANTIC_FEATURE_NAME],
                    "deep_residual": "deep",
                },
            ],
            "hidden_dim": int(FUSION_HIDDEN_DIM),
            "dropout": float(FUSION_DROPOUT),
            "batch_size": int(FUSION_BATCH_SIZE),
            "epochs": int(FUSION_EPOCHS),
            "patience": int(FUSION_PATIENCE),
            "lr": float(FUSION_LR),
            "weight_decay": float(FUSION_WEIGHT_DECAY),
            "cpc_loss_weight": float(FUSION_CPC_LOSS_WEIGHT),
            "fusion": "clinical_logit_plus_patient_specific_alpha_times_deep_delta",
        },
        "semantic_feature_name": SEMANTIC_FEATURE_NAME,
        "semantic_model": "PTBXL71",
        "semantic_output_dim": int(SEMANTIC_OUTPUT_DIM),
        "ptbxl_feed_mode": PTBXL_FEED_MODE,
        "ptbxl_target_len": int(PTBXL_TARGET_LEN),
        "ptbxl_crop_len": int(PTBXL_CROP_LEN),
        "ptbxl_crop_stride": int(PTBXL_CROP_STRIDE),
        "ptbxl_crop_agg": PTBXL_CROP_AGG,
        "stage1_backbone_config": {
            "name": backbone_key,
            "schedule": stage1_schedule_summary,
            "lr_summary": stage1_lr_summary,
            "ecgfounder_root": ECGFOUNDER_ROOT if backbone_key == "ecgfounder" else None,
            "ecgfounder_1lead_checkpoint": ECGFOUNDER_1LEAD_CKPT if backbone_key == "ecgfounder" else None,
            "seresnet_dropout": float(SERESNET_DROPOUT) if backbone_key == "seresnet" else None,
            "ecg_jepa_root": ECG_JEPA_ROOT if backbone_key == "ecg_jepa" else None,
            "ecg_jepa_checkpoint": ECG_JEPA_CKPT if backbone_key == "ecg_jepa" else None,
            "ecg_jepa_target_len": int(ECG_JEPA_TARGET_LEN) if backbone_key == "ecg_jepa" else None,
            "ecg_jepa_leads": list(ECG_JEPA_LEADS) if backbone_key == "ecg_jepa" else None,
            "ecgfm_root": ECGFM_ROOT if backbone_key == "ecgfm" else None,
            "ecgfm_checkpoint": ECGFM_CKPT if backbone_key == "ecgfm" else None,
            "ecgfm_target_len": int(ECGFM_TARGET_LEN) if backbone_key == "ecgfm" else None,
            "ecgfm_single_lead_index": int(ECGFM_SINGLE_LEAD_INDEX) if backbone_key == "ecgfm" else None,
            "stmem_root": ST_MEM_ROOT if backbone_key == "stmem" else None,
            "stmem_checkpoint": ST_MEM_CKPT if backbone_key == "stmem" else None,
            "stmem_target_len": int(ST_MEM_TARGET_LEN) if backbone_key == "stmem" else None,
            "stmem_patch_size": int(ST_MEM_PATCH_SIZE) if backbone_key == "stmem" else None,
            "stmem_single_lead_index": int(ST_MEM_SINGLE_LEAD_INDEX) if backbone_key == "stmem" else None,
        },
        "direct_val_loss": float(val_direct_loss),
        "direct_val_auroc": float(val_direct_auroc),
        "direct_val_auprc": float(val_direct_auprc),
        "direct_test_loss": float(test_direct_loss),
        "direct_test_auroc": float(test_direct_auroc),
        "direct_test_auprc": float(test_direct_auprc),
        "variants": sorted([f"{variant}::{rep}" for variant, rep in evaluated_design_specs]),
        "num_runs": int(NUM_RUNS),
    }
    with open(TRAIN_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] Raw: {TRAIN_RAW_CSV}")
    print(f"[Save] Summary: {TRAIN_SUMMARY_CSV}")
    print(f"[Save] Config: {TRAIN_CONFIG_JSON}")
    print(f"\n[Completed] Total time: {(time.time() - start_total) / 60:.1f} min")
