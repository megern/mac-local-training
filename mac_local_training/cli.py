"""Plan and run bounded MLX training on local Apple Silicon."""
import argparse
import json
from pathlib import Path
from .planner import estimate, estimate_adapter_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Show the raw-weight lower bound; no model is loaded")
    plan.add_argument("--parameters-billions", type=float, required=True)
    plan.add_argument("--bits", type=int, default=4)
    plan.add_argument("--memory-budget-gib", type=float, default=6)
    adapters = sub.add_parser("plan-adapters", help="Model raw FP32 adapter-state components without loading a model")
    adapters.add_argument("--trainable-parameters", type=int, required=True)
    adapters.add_argument("--active-fraction", type=float, default=1.0)
    train = sub.add_parser("train", help="Train a local model or the tiny software-check model")
    choice = train.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", type=Path)
    choice.add_argument("--tiny", action="store_true")
    train.add_argument("--train-data", type=Path, required=True)
    train.add_argument("--valid-data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--strategy", choices=["all", "rotating"], default="rotating")
    train.add_argument("--steps", type=int, default=12)
    train.add_argument("--microbatch-size", type=int, default=1)
    train.add_argument("--accumulation-steps", type=int, default=2)
    train.add_argument("--window-layers", type=int, default=1)
    train.add_argument("--window-updates", type=int, default=1)
    train.add_argument("--rank", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--max-seq-length", type=int, default=128)
    train.add_argument("--memory-budget-gib", type=float, default=6)
    train.add_argument("--checkpoint-every", type=int, default=1)
    train.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "plan-adapters":
            result = estimate_adapter_state(args.trainable_parameters, args.active_fraction)
        elif args.command == "plan":
            result = estimate(args.parameters_billions, args.bits, args.memory_budget_gib)
        else:
            from .engine import TrainingConfig, run_training
            config = TrainingConfig(train_path=str(args.train_data), valid_path=str(args.valid_data),
                output_path=str(args.output), model_path=str(args.model) if args.model else None,
                tiny_model=args.tiny, strategy=args.strategy, steps=args.steps,
                microbatch_size=args.microbatch_size, accumulation_steps=args.accumulation_steps,
                window_layers=args.window_layers, window_updates=args.window_updates,
                learning_rate=args.learning_rate, rank=args.rank, max_seq_length=args.max_seq_length,
                checkpoint_every=args.checkpoint_every, memory_budget_gib=args.memory_budget_gib,
                resume=args.resume)
            result = run_training(config)
            result = {key: value for key, value in result.items() if key not in {"history", "identity"}}
    except (ValueError, OSError, ImportError, FloatingPointError) as exc:
        parser.exit(2, f"Local run error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
