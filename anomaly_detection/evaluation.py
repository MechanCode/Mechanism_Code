import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .data import gather_segments, validate_boundaries


@torch.inference_mode()
def timestamp_scores(model, values, length, batch_size, device, boundaries=None):
    if boundaries is not None:
        validate_boundaries(boundaries, len(values))
        return np.concatenate([timestamp_scores(model, values[start:stop], length, batch_size, device)
                               for _, start, stop in boundaries])
    if len(values) < length:
        raise ValueError("Test sequence is shorter than one complete segment")
    model.eval()
    totals = np.zeros(len(values), dtype=np.float64)
    counts = np.zeros(len(values), dtype=np.int64)
    candidates = len(values) - length + 1
    for start in range(0, candidates, batch_size):
        ids = np.arange(start, min(start + batch_size, candidates))
        segments = torch.from_numpy(gather_segments(values, ids, length, 1)).to(device)
        errors = (model(segments) - segments).square().mean(dim=2).cpu().numpy()
        timestamps = ids[:, None] + np.arange(length)[None, :]
        np.add.at(totals, timestamps.ravel(), errors.ravel())
        np.add.at(counts, timestamps.ravel(), 1)
    if (counts == 0).any() or not np.isfinite(totals).all():
        raise FloatingPointError("Invalid or incomplete test scores")
    return totals / counts


def evaluate(model, values, labels, config, device, boundaries=None):
    if config.model_kind in ("feature_ae", "feature_linear"):


        from .model import BottleneckAutoencoder, LinearFeatureAutoencoder
        valid = ((config.model_kind == "feature_ae" and isinstance(model, BottleneckAutoencoder) and not model.temporal)
                 or (config.model_kind == "feature_linear" and isinstance(model, LinearFeatureAutoencoder)))
        if not valid:
            raise ValueError("Pointwise evaluation requires the feature autoencoder")
        if boundaries is not None:
            validate_boundaries(boundaries, len(values))
            if any(stop-start < config.segment_length for _, start, stop in boundaries):
                raise ValueError("Every evaluation sequence needs a complete segment")
        elif len(values) < config.segment_length:
            raise ValueError("Test sequence is shorter than one complete segment")
        model.eval()
        blocks = []
        with torch.inference_mode():
            chunk = config.eval_batch_size * config.segment_length
            for start in range(0, len(values), chunk):
                x = torch.from_numpy(values[start:start+chunk]).unsqueeze(0).to(device)
                blocks.append((model(x)-x).square().mean(dim=2).squeeze(0).cpu().numpy())
        scores = np.concatenate(blocks).astype(np.float64)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Invalid pointwise test scores")
    else:
        scores = timestamp_scores(model, values, config.segment_length, config.eval_batch_size, device, boundaries)
    metrics = {"auprc_ap": float(average_precision_score(labels, scores)),
               "auroc": float(roc_auc_score(labels, scores))}
    if config.f1_threshold is not None:
        metrics["f1"] = float(f1_score(labels, scores > config.f1_threshold, zero_division=0))
        metrics["f1_threshold"] = config.f1_threshold
    return scores, metrics
