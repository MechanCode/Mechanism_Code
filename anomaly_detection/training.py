from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from .data import segment_starts
from .evaluation import evaluate
from .io_utils import atomic_json, epsilon_tag, fingerprint
from .model import initialize_model
from .sampling import BatchSampler, pss_partition


class PrivateStep:


    def __init__(self, model, optimizer, config, noise_generator):
        self.model, self.optimizer, self.config = model, optimizer, config
        self.noise_generator = noise_generator

        def loss(parameters, buffers, segment):
            reconstruction = functional_call(model, (parameters, buffers), (segment.unsqueeze(0),))
            return config.loss_scale * (reconstruction.squeeze(0) - segment).square().mean()

        self.gradient = vmap(grad(loss), in_dims=(None, None, 0), randomness="error")

    def __call__(self, segments):
        config = self.config
        parameters = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        summed = {name: torch.zeros_like(p) for name, p in parameters.items()}
        for start in range(0, len(segments), config.microbatch_size):
            block = segments[start:start + config.microbatch_size]
            gradients = self.gradient(parameters, buffers, block)
            norm_squared = sum(g.detach().reshape(len(block), -1).square().sum(1)
                               for g in gradients.values())
            if not torch.isfinite(norm_squared).all():
                raise FloatingPointError("Non-finite per-segment gradient; run was not completed")
            factors = (config.clipping_norm / norm_squared.sqrt().clamp_min(1e-30)).clamp(max=1)
            for name, g in gradients.items():
                weights = factors.reshape((-1,) + (1,) * (g.ndim - 1))
                summed[name].add_((g.detach() * weights).sum(0))
        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in parameters.items():


            noise = torch.randn(parameter.shape, dtype=parameter.dtype, device="cpu",
                                generator=self.noise_generator).to(parameter.device)
            parameter.grad = (summed[name] + noise * config.clipping_norm * config.noise_multiplier) / config.batch_size
        self.optimizer.step()


def make_optimizer(model, config):

    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                               momentum=config.momentum, weight_decay=0)
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                betas=(config.adam_beta1, config.adam_beta2),
                                eps=config.adam_epsilon, weight_decay=0)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def environment_identity(device):
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
              "numpy": np.__version__, "device": str(device),
              "threads": torch.get_num_threads(), "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version()}
    for name in ("scipy", "pandas", "scikit-learn", "dp-accounting"):
        result[name] = importlib.metadata.version(name)
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
    return result


def resolve_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu or cuda[:index]")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def atomic_checkpoint(path, state):
    descriptor, temporary = tempfile.mkstemp(prefix="checkpoint.", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is running {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_experiment(config, plan, training, test, labels, labels_hash, output_dir,
                   method, epsilon, seed, *, device="auto", resume=False,
                   save_scores=False, progress=print, stop_after=None):

    epsilon = float(epsilon)
    if method not in plan["methods"] or epsilon not in config.epsilons:
        raise ValueError("Method/epsilon is absent from the accounted plan")
    if not config.seed_start <= seed < config.seed_start + config.num_seeds:
        raise ValueError("Seed is outside the configured repetition range")
    if plan["protocol"] != config.to_dict() or plan["training"] != training.identity():
        raise ValueError("Training data/configuration differs from the accounted plan")
    budget = next(b for b in plan["methods"][method]["budgets"] if b["target_epsilon"] == epsilon)
    if not budget["budget_exhausted"] and not config.allow_step_cap:
        raise ValueError("This plan stops at a computation cap, not a privacy boundary")
    device = resolve_device(device)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    environment = environment_identity(device)
    manifest = {"plan_id": fingerprint(plan), "method": method, "epsilon": epsilon,
                "seed": seed, "test": test.identity(), "labels_sha256": labels_hash,
                "environment": environment}
    identity = fingerprint(manifest)
    directory = Path(output_dir) / method / f"epsilon_{epsilon_tag(epsilon)}" / f"seed_{seed}"
    with run_lock(directory):
        result_path, checkpoint_path = directory / "result.json", directory / "checkpoint.pt"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["run_id"] != identity:
                raise ValueError(f"Existing result has different data/config/code/environment: {directory}")
            progress(f"Already complete: {method}, epsilon={epsilon:g}, seed={seed}")
            return result
        if checkpoint_path.exists() and not resume:
            raise FileExistsError(f"Checkpoint exists; use --resume: {checkpoint_path}")
        model = initialize_model(training.values.shape[1], seed, config).to(device)
        model.train()
        optimizer = make_optimizer(model, config)
        sample_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        noise_seed = int(np.random.SeedSequence([seed, 2]).generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63))
        noise_rng = torch.Generator(device="cpu").manual_seed(noise_seed)
        partition = (pss_partition(plan["geometry"]["candidates"], config.batch_size,
                                   budget["lambda"], config.pss_mode) if method == "pss" else None)
        sampler = BatchSampler(method, plan["geometry"]["candidates"], config.batch_size, sample_rng, partition)
        starts = segment_starts(training, config.segment_length, config.train_stride)
        if len(starts) != plan["geometry"]["candidates"]:
            raise ValueError("Accounted and actual training window counts differ")
        step = 0
        if checkpoint_path.exists():

            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if state["run_id"] != identity:
                raise ValueError("Checkpoint identity differs; resume would change the experiment")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            sample_rng.bit_generator.state = state["sampling_rng"]
            noise_rng.set_state(state["noise_rng"])
            step = state["step"]
            if not 0 <= step <= budget["steps"]:
                raise ValueError("Invalid checkpoint step")
        atomic_json(directory / "manifest.json", {**manifest, "run_id": identity, "budget": budget})
        private_step = PrivateStep(model, optimizer, config, noise_rng)

        def checkpoint():
            atomic_checkpoint(checkpoint_path, {"run_id": identity, "step": step,
                              "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "sampling_rng": sample_rng.bit_generator.state,
                              "noise_rng": noise_rng.get_state()})

        progress(f"Training {method}, epsilon={epsilon:g}, seed={seed}: {step}/{budget['steps']} steps")
        final_step = budget["steps"] if stop_after is None else min(stop_after, budget["steps"])
        while step < final_step:
            ids = sampler.draw()
            segments = training.values[starts[ids, None] + np.arange(config.segment_length)[None, :]]
            private_step(torch.from_numpy(segments).to(device))
            step += 1
            if step % config.checkpoint_every == 0:
                checkpoint()
                progress(f"  {method}, epsilon={epsilon:g}, seed={seed}: {step}/{budget['steps']}")
        checkpoint()
        if step < budget["steps"]:
            return None
        scores, metrics = evaluate(model, test.values, labels, config, device, test.sequence_boundaries)
        if save_scores:

            np.savez_compressed(directory / "scores.npz", scores=scores, labels=labels,
                                timestamps=test.timestamps)
        result = {**manifest, "run_id": identity, "status": "complete", "budget": budget,
                  "steps": step, "training_performed": step > 0, "metrics": metrics}
        atomic_json(result_path, result)
        progress(f"Finished {method}, epsilon={epsilon:g}, seed={seed}: "
                 f"AP={metrics['auprc_ap']:.6f}, AUROC={metrics['auroc']:.6f}")
        return result
