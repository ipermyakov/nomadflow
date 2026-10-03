from pathlib import Path
import argparse
import logging
import os
import shutil
import sys
import types
import typing as T
from timeit import default_timer as timer

# fair-esm[esmfold] pins deepspeed==0.5.9, which imports torch._six (removed in torch>=2.0).
# OpenFold probes deepspeed at import time; shim the module before pulling in esm/openfold.
if "torch._six" not in sys.modules:
    _torch_six = types.ModuleType("torch._six")
    setattr(_torch_six, "inf", float("inf"))
    sys.modules["torch._six"] = _torch_six

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.custom_(fwd|bwd)\(args\.\.\.\)` is deprecated",
    category=FutureWarning,
)

import torch

import esm
from esm.data import read_fasta

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


PathLike = T.Union[str, Path]


def fasta_length_sort_key(path: str) -> float:
    """Sort PMPNN FASTA files by the ``len_<N>`` token in the basename.

    Unprefixed names (two-task slice): ``len_76_protein_id0_0_0.fa``.
    Shard-prefixed names (merged full/se3): ``s0_1qjg__len_100_protein_id0_25_0.fa``.

    ``float(basename.split("_")[1])`` only works for the unprefixed form and
    raises ``ValueError`` on shard prefixes such as ``7k4v`` or ``short``.
    """
    name = os.path.basename(path)
    try:
        return float(name.split("len_", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse length from FASTA filename: {name!r}") from exc


ESMFOLD_MODEL_NAME = "esmfold_3B_v1"
ESMFOLD_MODEL_URL = (
    f"https://dl.fbaipublicfiles.com/fair-esm/models/{ESMFOLD_MODEL_NAME}.pt"
)
ESMFOLD_DOWNLOAD_USER_AGENT = "Mozilla/5.0 (compatible; nomadflow-esmfold/1.0)"
DEFAULT_ESMFOLD_MODEL_PATH = Path(
    os.environ.get(
        "ESMFOLD_MODEL_PATH",
        str(Path.home() / "progs/esmfold_weights/esmfold_3B_v1.pt"),
    )
)
ESM2_MODEL_NAME = "esm2_t36_3B_UR50D"
ESM2_MODEL_URL = f"https://dl.fbaipublicfiles.com/fair-esm/models/{ESM2_MODEL_NAME}.pt"
ESM2_REGRESSION_URL = (
    f"https://dl.fbaipublicfiles.com/fair-esm/regression/{ESM2_MODEL_NAME}-contact-regression.pt"
)
DEFAULT_ESM2_MODEL_PATH = Path(
    os.environ.get(
        "ESM2_MODEL_PATH",
        str(Path.home() / "progs/esmfold_weights/esm2_t36_3B_UR50D.pt"),
    )
)
_OPENFOLD_LINEAR_WRAPPED_PREFIXES = (
    "trunk.structure_module.ipa.linear_q_points",
    "trunk.structure_module.ipa.linear_kv_points",
)


def _normalize_esmfold_checkpoint_keys(checkpoint: T.Any) -> T.Any:
    """Adapt Meta ESMFold v1 weights to newer OpenFold Linear module names."""
    if not isinstance(checkpoint, dict):
        return checkpoint

    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        return checkpoint

    patched = 0
    for prefix in _OPENFOLD_LINEAR_WRAPPED_PREFIXES:
        for suffix in ("weight", "bias"):
            old_key = f"{prefix}.{suffix}"
            new_key = f"{prefix}.linear.{suffix}"
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict[old_key]
                patched += 1

    if patched:
        logger.info("Patched %d ESMFold checkpoint keys for current OpenFold.", patched)

    return checkpoint


def _patch_openfold_deepspeed_comm() -> None:
    try:
        import openfold.model.primitives as primitives
    except Exception:
        return

    deepspeed = getattr(primitives, "deepspeed", None)
    try:
        getattr(getattr(deepspeed, "comm"), "comm").is_initialized
    except Exception:
        if getattr(primitives, "deepspeed_is_installed", False):
            primitives.deepspeed_is_installed = False
            logger.info("Disabled OpenFold deepspeed path for incompatible deepspeed API.")


def download_esm_asset(dest: Path, url: str) -> None:
    """Download ESMFold checkpoint; Meta CDN returns 403 without a User-Agent."""
    import shutil
    import subprocess
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")

    if shutil.which("curl"):
        cmd = [
            "curl",
            "-fL",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-A",
            ESMFOLD_DOWNLOAD_USER_AGENT,
            "-o",
            str(tmp),
            url,
        ]
        if dest.exists() and dest.stat().st_size > 0:
            cmd.insert(1, "-C")
            cmd.insert(2, "-")
        subprocess.run(cmd, check=True)
        tmp.replace(dest)
        return

    request = urllib.request.Request(url, headers={"User-Agent": ESMFOLD_DOWNLOAD_USER_AGENT})
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)


def download_esmfold_weights(dest: Path, url: str = ESMFOLD_MODEL_URL) -> None:
    download_esm_asset(dest, url)


def _esm2_regression_path(esm2_path: Path) -> Path:
    return esm2_path.with_name(f"{esm2_path.stem}-contact-regression.pt")


def ensure_esm2_weights(esm2_path: Path = DEFAULT_ESM2_MODEL_PATH) -> Path:
    esm2_path = Path(os.environ.get("ESM2_MODEL_PATH", str(esm2_path)))
    if not esm2_path.exists() or esm2_path.stat().st_size == 0:
        logger.info("ESM-2 checkpoint missing at %s; downloading (~5.3 GB)", esm2_path)
        download_esm_asset(esm2_path, ESM2_MODEL_URL)

    regression_path = _esm2_regression_path(esm2_path)
    if not regression_path.exists() or regression_path.stat().st_size == 0:
        logger.info("Downloading ESM-2 contact-regression weights to %s", regression_path)
        download_esm_asset(regression_path, ESM2_REGRESSION_URL)

    return esm2_path


def _patch_esm2_pretrained_local(esm2_path: Path) -> None:
    import esm.pretrained as esm_pretrained

    def esm2_t36_3B_UR50D_local():
        return esm_pretrained.load_model_and_alphabet_local(str(esm2_path))

    esm_pretrained.esm2_t36_3B_UR50D = esm2_t36_3B_UR50D_local


def _load_esmfold_checkpoint(checkpoint: str):
    from esm.esmfold.v1.pretrained import _load_model

    _patch_openfold_deepspeed_comm()
    esm2_path = ensure_esm2_weights()

    # PyTorch 2.6+: torch.load defaults to weights_only=True; ESMFold checkpoints
    # contain OmegaConf DictConfig. Official Meta weights are a trusted source.
    _orig_torch_load = torch.load

    def _torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        loaded = _orig_torch_load(*args, **kwargs)
        return _normalize_esmfold_checkpoint_keys(loaded)

    torch.load = _torch_load
    _patch_esm2_pretrained_local(esm2_path)
    try:
        logger.info("Loading ESM-2 backbone from %s", esm2_path)
        return _load_model(checkpoint)
    finally:
        torch.load = _orig_torch_load


def load_esmfold_model(
    model_dir: T.Optional[PathLike] = None,
    model_path: T.Optional[PathLike] = None,
):
    if model_path is None:
        model_path = os.environ.get("ESMFOLD_MODEL_PATH", str(DEFAULT_ESMFOLD_MODEL_PATH))

    if model_path is not None:
        checkpoint = Path(model_path)
        if not checkpoint.exists() or checkpoint.stat().st_size == 0:
            logger.info("ESMFold checkpoint missing at %s; downloading", checkpoint)
            download_esmfold_weights(checkpoint, ESMFOLD_MODEL_URL)
        logger.info("Loading ESMFold from %s", checkpoint)
        t0 = timer()
        logger.info("Reading checkpoint from disk (can take several minutes)...")
        model = _load_esmfold_checkpoint(str(checkpoint))
        logger.info("ESMFold checkpoint loaded in %.1fs", timer() - t0)
        return model

    if model_dir is not None:
        torch.hub.set_dir(str(model_dir))

    hub_checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    hub_checkpoints.mkdir(parents=True, exist_ok=True)
    cached = hub_checkpoints / f"{ESMFOLD_MODEL_NAME}.pt"

    if cached.exists() and cached.stat().st_size > 0:
        logger.info("Loading cached ESMFold weights from %s", cached)
        return _load_esmfold_checkpoint(str(cached))

    for partial in hub_checkpoints.glob(f"{ESMFOLD_MODEL_NAME}.pt.*.partial"):
        if partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)

    logger.info(
        "Downloading ESMFold weights (~2.6 GB) to %s. "
        "First run can take several minutes; progress bar below.",
        cached,
    )
    download_esmfold_weights(cached, ESMFOLD_MODEL_URL)
    return _load_esmfold_checkpoint(str(cached))


def enable_cpu_offloading(model):
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:9999", world_size=1, rank=0
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--fasta_folder",
        help="Path to folder of input FASTA file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o", "--pdb", help="Path to output PDB directory", type=Path, required=True
    )
    parser.add_argument(
        "-m", "--model-dir", help="Parent path to Pretrained ESM data directory. ", type=Path, default=None
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1500,
        help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
        "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
        "short sequences.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). "
        "Equivalent to running a for loop over chunks of of each dimension. Lower values will "
        "result in lower memory usage at the cost of speed. Recommended values: 128, 64, 32. "
        "Default: None.",
    )
    parser.add_argument("--cpu-only", help="CPU only", action="store_true")
    parser.add_argument("--cpu-offload", help="Enable CPU offloading", action="store_true")
    parser.add_argument("--force", help="overwrite existing folded output folders", action="store_true")
    args = parser.parse_args()

    args.pdb.mkdir(exist_ok=True)

    logger.info("Loading model")

    model = load_esmfold_model(model_dir=args.model_dir)


    model = model.eval()
    model.set_chunk_size(args.chunk_size)

    if args.cpu_only:
        model.esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
        model.cpu()
    elif args.cpu_offload:
        model = init_model_on_gpu_with_cpu_offloading(model)
    else:
        logger.info("Moving ESMFold to GPU...")
        model.cuda()
        logger.info("ESMFold ready on %s", next(model.parameters()).device)

    import glob

    for fasta in sorted(
        glob.glob(os.path.join(args.fasta_folder, "*.fa")),
        key=fasta_length_sort_key,
    ):
        os.chdir(args.pdb)
        fasta = Path(fasta)

        if not fasta.exists():
            raise FileNotFoundError(fasta)

        sample_dir = os.path.splitext(os.path.basename(fasta))[0]
        if os.path.exists(sample_dir) and args.force:
            logger.info("Removing existing %s because --force was set", sample_dir)
            shutil.rmtree(sample_dir)

        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        else:
            logger.info(f"Skipping {sample_dir} because folder already exists")
            continue

        logger.info("Starting Predictions")

        # Read fasta and sort sequences by length
        logger.info(f"Reading sequences from {fasta}")
        all_sequences = sorted(read_fasta(fasta), key=lambda header_seq: len(header_seq[1]))
        logger.info(f"Loaded {len(all_sequences)} sequences from {fasta}")

        batched_sequences = create_batched_sequence_datasest(all_sequences, args.max_tokens_per_batch)

        num_completed = 0
        num_sequences = len(all_sequences)
        for headers, sequences in batched_sequences:
            start = timer()
            try:
                output = model.infer(sequences, num_recycles=args.num_recycles)
            except RuntimeError as e:
                if e.args[0].startswith("CUDA out of memory"):
                    if len(sequences) > 1:
                        logger.info(
                            f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                            "Try lowering `--max-tokens-per-batch`."
                        )
                    else:
                        logger.info(
                            f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}."
                        )

                    continue
                raise

            output = {key: value.cpu() for key, value in output.items()}
            pdbs = model.output_to_pdb(output)
            tottime = timer() - start
            time_string = f"{tottime / len(headers):0.1f}s"
            if len(sequences) > 1:
                time_string = time_string + f" (amortized, batch size {len(sequences)})"
            for header, seq, pdb_string, mean_plddt, ptm, pae in zip(
                headers, sequences, pdbs, output["mean_plddt"], output["ptm"], output["aligned_confidence_probs"]
            ):
                pae = (pae.cpu() * torch.arange(64)).mean(-1) * 31
                pae = pae.mean().numpy(force=True)

                output_file = args.pdb / sample_dir / f"{header}.pdb"
                output_file.write_text(pdb_string)
                num_completed += 1
                logger.info(
                    f"Predicted structure for {header} with length {len(seq)}, pLDDT {mean_plddt:0.1f}, "
                    f"pTM {ptm:0.3f}, pAE {pae:0.3f} in {time_string}. "
                    f"{num_completed} / {num_sequences} completed."
                )
