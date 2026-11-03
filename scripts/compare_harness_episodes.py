"""Run fixed Git snapshots on paired HM3D episodes and retain failed attempts.

模型/镜像/数据配置相同；每个 episode 独立进程，一次工程失败保留为该样本结果。
baseline 使用原策略；refactor 使用显式 agent 模式，因此比较包含策略变化。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    """Freeze two revisions, execute paired episodes serially, and export results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--refactor-ref', default='HEAD')
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--model', default='doubao-seed-2-0-pro-260215')
    parser.add_argument('--agent-max-model-calls', type=int, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    key = next(line.strip() for line in args.key_file.read_text().splitlines() if line.strip().startswith('ark-'))
    refs = {'baseline': args.baseline_ref, 'refactor': args.refactor_ref}
    commits = {}
    for arm, ref in refs.items():
        commit = subprocess.check_output(['git', 'rev-parse', ref], cwd=root, text=True).strip()
        commits[arm] = commit
        destination = args.output / arm
        destination.mkdir()
        archive = destination / 'source.tar'
        subprocess.run(['git', 'archive', '-o', str(archive), commit], cwd=root, check=True)
        source = destination / 'source'
        source.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(source)
    config = {'commits': commits, 'model': args.model, 'episodes': list(range(args.episodes)),
              'max_steps': 500, 'dataset_episodes': args.episodes, 'image': 'strive-hm3d:local',
              'refactor_decision_owner': 'agent', 'agent_max_model_calls': args.agent_max_model_calls,
              'weights_config_sha256': hashlib.sha256((root / 'configs/strive_weights.yaml').read_bytes()).hexdigest()}
    (args.output / 'comparison-config.json').write_text(json.dumps(config, indent=2))
    results = []
    for index in range(args.episodes):
        for arm in refs:
            source = args.output / arm / 'source'
            output = args.output / arm / f'episode-{index}'
            output.mkdir()
            save_dir = f'paired_{arm}_{index}'
            env = os.environ.copy()
            env.update(ARK_API_KEY=key, STRIVE_ROOT=str(source),
                       STRIVE_WEIGHTS_CONFIG=str(root / 'configs/strive_weights.yaml'),
                       CONTAINER_NAME=f'strive-paired-{arm}-{index}', LLM_MODEL=args.model,
                       STRIVE_LLM_CLIENT='ark', LLM_OFFLINE='0', STRIVE_LLM_FALLBACK='0',
                       STRIVE_LLM_PARSE_FALLBACK='0', PYTHONHASHSEED='0')
            command = ['bash', str(source / 'docker/run_hm3d_baseline.sh'),
                       '--eval_episodes', str(index + 1), '--start_episode', str(index),
                       '--dataset_episodes', str(args.episodes), '--max_steps', '500',
                       '--save_dir', save_dir, '--run_id', save_dir, '--vlm', 'ark']
            if arm == 'refactor':
                command += ['--decision_owner', 'agent', '--agent_max_model_calls', str(args.agent_max_model_calls)]
            (output / 'command.json').write_text(json.dumps(command, indent=2))
            started = time.monotonic()
            with (output / 'episode.log').open('w') as stream:
                process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            result = {'arm': arm, 'episode_index': index, 'commit': commits[arm],
                      'returncode': process.returncode, 'elapsed_s': time.monotonic() - started}
            metrics_file = source / 'logs' / save_dir / 'metrics.csv'
            if metrics_file.is_file():
                with metrics_file.open() as stream:
                    result['metrics'] = list(csv.DictReader(stream))
            frames = source / 'logs' / save_dir / f'episode-{index}' / 'rgb'
            result['initial_rgb_sha256'] = {
                str(step): hashlib.sha256((frames / f'monitor-rgb_{step}.jpg').read_bytes()).hexdigest()
                for step in (1, 2, 3) if (frames / f'monitor-rgb_{step}.jpg').is_file()
            }
            (output / 'result.json').write_text(json.dumps(result, indent=2))
            results.append(result)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2))
            print(f'{arm} episode={index} exit={process.returncode} elapsed={result["elapsed_s"]:.1f}s', flush=True)
    print(f'Completed {len(results)} attempts; results: {args.output / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
