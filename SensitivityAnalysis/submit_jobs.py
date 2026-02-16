"""SLURM job orchestration for ISP sensitivity analysis.

Discovers all ISP variants under the data directory and generates/submits
one SLURM job per variant that trains all models in parallel on separate GPUs::

    python -m SensitivityAnalysis.submit_jobs \\
        --data-root /storage/data \\
        --output-root /results/sensitivity \\
        --models yolov8m yolo26m \\
        --dry-run
"""

import argparse
import logging
import subprocess
import textwrap
from pathlib import Path
from typing import List, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_ROOT = "/data/Vijayakumar.TejusVignesh/GMIND/"
OUTPUT_ROOT = "/data/Vijayakumar.TejusVignesh/sensitivity_results/"

DEFAULT_MODELS = ["yolov8m", "yolo26m", "fasterrcnn_resnet50_fpn", "rtdetr-l"]

MODEL_DEFAULTS = {
    "yolov8m":                 {"batch_size": 64, "backend": "ultralytics", "epochs": 75, "lr": 0.001},
    "yolo26m":                 {"batch_size": 64, "backend": "ultralytics", "epochs": 75, "lr": 0.001},
    "fasterrcnn_resnet50_fpn": {"batch_size": 8,  "backend": "torchvision", "epochs": 30, "lr": 0.005},
    "rtdetr-l":                {"batch_size": 32, "backend": "ultralytics", "epochs": 100, "lr": 0.0001},
}


# ---------------------------------------------------------------------------
# Variant discovery
# ---------------------------------------------------------------------------

def discover_variants(
    data_root: Path,
    sets: List[str],
    subdirs: List[int],
    sensor: str,
) -> List[str]:
    """Walk the data tree and return a sorted list of unique ISP variant names."""
    variants: Set[str] = set()

    for set_name in sets:
        for sd in subdirs:
            proc_dir = data_root / set_name / str(sd) / "Processed_Images"
            if not proc_dir.exists():
                continue

            # Check {sensor}/<variant>/<variant>.mp4
            sensor_dir = proc_dir / sensor
            if sensor_dir.is_dir():
                for child in sensor_dir.iterdir():
                    if not child.is_dir():
                        continue
                    # Variant dir should contain a .mp4
                    if list(child.glob("*.mp4")):
                        variants.add(child.name)

            # Check Bayer_GC/
            bayer_dir = proc_dir / "Bayer_GC"
            if bayer_dir.is_dir() and list(bayer_dir.glob("*.mp4")):
                variants.add("Bayer_GC")

    result = sorted(variants)
    logger.info(f"Discovered {len(result)} ISP variants: {result}")
    return result


# ---------------------------------------------------------------------------
# SLURM script generation
# ---------------------------------------------------------------------------

def generate_slurm_script(
    variant: str,
    models: List[str],
    output_dir: str,
    partition: str,
    gpus: int,
    cpus_per_task: int,
    mail_user: str,
    config_path: str,
    num_workers: int,
    cli_lr: float,
) -> str:
    """Return the content of a SLURM batch script.

    Each model runs on a dedicated GPU via CUDA_VISIBLE_DEVICES, all in
    parallel as background processes joined by a final ``wait``.
    """
    job_name = f"train_{variant}"

    header = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output=logs/{job_name}_%j.out
        #SBATCH --error=logs/{job_name}_%j.err
        #SBATCH --partition={partition}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={cpus_per_task}
        #SBATCH --gres=gpu:{gpus}
        #SBATCH --mail-user={mail_user}
        #SBATCH --mail-type=ALL
        module load miniconda/2405
    """)

    blocks = []
    for gpu_idx, model in enumerate(models):
        defaults = MODEL_DEFAULTS.get(model, {})
        batch_size = defaults.get("batch_size", 32)
        backend = defaults.get("backend", "auto")
        epochs = defaults.get("epochs", 50)
        lr = defaults.get("lr", cli_lr)

        block = (
            f"CUDA_VISIBLE_DEVICES={gpu_idx} conda run -n gmind "
            f"python -m DeepLearning.train_models \\\n"
            f"    --use-gmind \\\n"
            f"    --gmind-config {config_path} \\\n"
            f"    --isp-variant {variant} \\\n"
            f"    --model {model} \\\n"
            f"    --backend {backend} \\\n"
            f"    --epochs {epochs} \\\n"
            f"    --batch-size {batch_size} \\\n"
            f"    --lr {lr} \\\n"
            f"    --num-workers {num_workers} \\\n"
            f"    --do-eval \\\n"
            f"    --checkpoint-dir {output_dir} \\\n"
            f"    --device cuda &"
        )
        blocks.append(block)

    return header + "\n" + "\n\n".join(blocks) + "\n\nwait\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Discover ISP variants and submit SLURM training jobs"
    )
    p.add_argument("--data-root", default=DATA_ROOT, help="Root data directory")
    p.add_argument("--output-root", default=OUTPUT_ROOT, help="Root output directory")

    p.add_argument("--sensor", default="FLIR8.9")
    p.add_argument("--sets", nargs="+", default=["NightUrbanJunction"])
    p.add_argument("--subdirs", nargs="+", type=int, default=[1, 2])
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)

    p.add_argument("--partition", default="gpuq")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--cpus-per-task", type=int, default=40)
    p.add_argument("--mail-user", default="vijayakumar.tejusvignesh@ul.ie")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--num-workers", type=int, default=8, help="Number of data loader workers")

    p.add_argument(
        "--variants", nargs="+", default=None,
        help="Specific variants (default: auto-discover)",
    )
    p.add_argument(
        "--config", default=str(Path(__file__).parent / "sensitivity_config.yaml"),
        help="Path to sensitivity config YAML",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    p.add_argument("--generate-only", action="store_true", help="Write scripts without submitting")
    return p


def main():
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    # Discover variants
    if args.variants:
        variants = args.variants
        logger.info(f"Using user-specified variants: {variants}")
    else:
        variants = discover_variants(data_root, args.sets, args.subdirs, args.sensor)
        if not variants:
            logger.error("No ISP variants discovered. Check --data-root and --sets.")
            return

    scripts_dir = Path(__file__).resolve().parent.parent / "slurm_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    total_jobs = len(variants)
    logger.info(
        f"Generating {total_jobs} jobs "
        f"({len(variants)} variants, {len(args.models)} models per job)"
    )

    submitted = 0
    for variant in variants:
        script_content = generate_slurm_script(
            variant=variant,
            models=args.models,
            output_dir=str(output_root),
            partition=args.partition,
            gpus=args.gpus,
            cpus_per_task=args.cpus_per_task,
            mail_user=args.mail_user,
            config_path=args.config,
            num_workers=args.num_workers,
            cli_lr=args.lr,
        )

        script_path = scripts_dir / f"{variant}.sh"
        script_path.write_text(script_content)

        if args.dry_run:
            logger.info(f"[DRY RUN] sbatch {script_path}")
        elif args.generate_only:
            logger.info(f"Generated: {script_path}")
        else:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                logger.info(f"Submitted: {result.stdout.strip()} — {variant}")
                submitted += 1
            else:
                logger.error(f"sbatch failed for {variant}: {result.stderr.strip()}")

    if not args.dry_run and not args.generate_only:
        logger.info(f"Submitted {submitted}/{total_jobs} jobs")
    elif args.generate_only:
        logger.info(f"Scripts written to {scripts_dir}")


if __name__ == "__main__":
    main()
