import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ccaa",
        type=int,
        required=True,
        help="Maximum runtime supplied by the evaluator.",
    )
    args = parser.parse_args()

    solution_dir = Path(__file__).resolve().parent
    data_dir = solution_dir / "data"
    artifacts_dir = solution_dir / "artifacts"

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"script={Path(__file__).name}")
    print(f"timeout_seconds={args.timeout_seconds}")
    print(f"data_directory={data_dir}")
    print(f"data_exists={data_dir.exists()}")
    print(f"artifacts_directory={artifacts_dir}")


if __name__ == "__main__":
    main()
