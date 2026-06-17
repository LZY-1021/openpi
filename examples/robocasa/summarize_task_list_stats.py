import argparse
import json
from pathlib import Path


def read_tasks(task_list_file: str) -> list[str]:
    tasks = []
    with open(task_list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            tasks.append(line.split()[0])
    return tasks


def latest_stats_path(result_dir: Path, split: str, task: str) -> Path | None:
    for eval_dir_name in ("evals_1.5", "evals"):
        task_dir = result_dir / eval_dir_name / split / task
        if not task_dir.exists():
            continue
        run_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())
        for run_dir in reversed(run_dirs):
            stats_path = run_dir / "stats.json"
            if stats_path.exists():
                return stats_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--task_list_file", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    tasks = read_tasks(args.task_list_file)

    total_episodes = 0
    total_successes = 0.0

    print(f"{'Task':40s} {'Episodes':>10s} {'Success Rate':>12s}")
    print("-" * 66)

    for task in tasks:
        stats_path = latest_stats_path(result_dir, args.split, task)
        if stats_path is None:
            print(f"{task:40s} {0:10d} {'-':>12s}")
            continue

        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        episodes = int(stats.get("num_episodes", 0))
        success_rate = float(stats.get("success_rate", 0.0))
        total_episodes += episodes
        total_successes += success_rate * episodes

        print(f"{task:40s} {episodes:10d} {success_rate:12.3f}")

    print("-" * 66)
    if total_episodes == 0:
        print(f"{'TOTAL':40s} {0:10d} {'-':>12s}")
    else:
        print(f"{'TOTAL':40s} {total_episodes:10d} {total_successes / total_episodes:12.3f}")


if __name__ == "__main__":
    main()
