import argparse
from pathlib import Path


def _cmd_prep(args):
    from trainer_desktop.core.data_prep import DataPrepConfig, prepare_dataset

    cfg = DataPrepConfig(
        input_dir=Path(args.input),
        runs_dir=Path(args.runs),
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
        run_name=args.name,
        dry_run=args.dry_run,
    )
    res = prepare_dataset(cfg)
    print("Run dir:", res.run_dir)
    print("Total images:", res.total_images)
    print(f"Split => train:{res.train} valid:{res.valid} test:{res.test}")
    if res.warnings:
        print("Warnings:")
        for w in res.warnings:
            print(" -", w)


def _cmd_yaml(args):
    from trainer_desktop.core.yaml_gen import generate_yaml
    run = Path(args.run)
    ds = Path(args.path) if args.path else None
    out = generate_yaml(run, dataset_root=ds)
    print("Wrote:", out)


def _cmd_gui(_args):
    from trainer_desktop.app import main
    main()


def main():
    p = argparse.ArgumentParser(description="Desktop Trainer Tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prep", help="Run data preparation")
    p_prep.add_argument("--input", required=True, help="Input folder containing images, labels and classes.txt")
    p_prep.add_argument("--runs", default="runs", help="Runs output folder")
    p_prep.add_argument("--train", type=float, default=0.7, help="Train ratio (0-1)")
    p_prep.add_argument("--val", type=float, default=0.2, help="Val ratio (0-1)")
    p_prep.add_argument("--seed", type=int, default=42, help="Random seed")
    p_prep.add_argument("--name", default=None, help="Run name (default: timestamp)")
    p_prep.add_argument("--dry-run", action="store_true", help="Do not write any files")
    p_prep.set_defaults(func=_cmd_prep)

    p_yaml = sub.add_parser("yaml", help="Generate data.yaml for a run")
    p_yaml.add_argument("--run", required=True, help="Run folder containing classes.txt and split files")
    p_yaml.add_argument("--path", help="Dataset root path to embed in YAML")
    p_yaml.set_defaults(func=_cmd_yaml)

    p_gui = sub.add_parser("gui", help="Launch Tkinter UI")
    p_gui.set_defaults(func=_cmd_gui)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
