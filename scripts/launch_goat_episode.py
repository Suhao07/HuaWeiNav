"""Launch one isolated GOAT parent episode with the local Qwen service."""
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.habitat_evaluation import strict_json_loads


def language_inputs(task_file, index, data_root, output, max_steps, max_model_calls):
    with gzip.open(task_file, 'rt') as stream:
        episodes = json.load(stream)['episodes']
    if not 0 <= index < len(episodes):
        raise ValueError('Language task index is outside the frozen subset')
    selected = episodes[index]
    if selected['info']['protocol'] != 'goat_language_single_parent_start_v1':
        raise ValueError('Expected the independent GOAT language subset')
    identities = [{**{key: row[key] for key in
                      ('episode_id', 'start_position', 'start_rotation', 'object_category')},
                   'scene_id': str(data_root / 'scene_datasets' / row['scene_id'])}
                  for row in episodes]
    manifest = output / 'runtime_episodes.json'
    manifest.write_text(json.dumps({'runtime_episodes': identities}) + '\n')
    argv = ['objnav_benchmark_with_process_obs.py', '--benchmark', 'goat_language',
            '--benchmark_split', 'val_unseen', '--dataset_path', str(task_file),
            '--dataset_episodes', str(len(episodes)), '--episode_manifest', str(manifest),
            '--start_episode', str(index), '--eval_episodes', str(index + 1),
            '--max_steps', str(max_steps),
            '--agent_max_model_calls', str(max_model_calls), '--vlm', 'self_hosted',
            '--save_dir', 'task']
    return argv, identities[index], selected['info']['instruction']


def language_score(output, index, expected_identity, instruction, independent):
    """Bind separately computed independent evidence while retaining raw benchmark metrics."""
    folder = output / 'navigation/task'
    with (folder / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or int(rows[0]['Episode']) != index:
        raise ValueError('Expected exactly the selected language task result')
    row = rows[0]
    actual = strict_json_loads((folder / f'episode-{index}/episode_identity.json').read_text())
    if actual != expected_identity or row['object_goal'] != instruction:
        raise ValueError('Executed task identity or instruction differs from frozen input')
    if row['benchmark'] != 'goat_language' or float(row['benchmark_success_distance']) != 0.25:
        raise ValueError('Language task scoring protocol differs')
    if row['End'] != '1':
        raise ValueError('Language task metrics are unfinished')
    provenance = independent['provenance']
    if provenance['schema'] != 'goat_language_viewpoints_v1':
        raise ValueError('Independent evaluation protocol differs')
    # Missing evidence may have no identity; any supplied identity must still match.
    if (independent['status'] == 'available' or 'episode_identity' in provenance):
        if provenance.get('episode_identity') != expected_identity:
            raise ValueError('Independent evaluation episode identity differs')
    metrics = {name: float(row[key]) for name, key in
               (('SR', 'success'), ('SPL', 'spl'), ('distance_to_goal', 'distance_to_goal'))}
    ready = all(math.isfinite(value) for value in metrics.values())
    score = {'aggregate_ready': ready, 'metrics': metrics if ready else None, 'raw_metrics': row,
             'independent_metrics': independent,
             'protocol': 'goat_language_single_parent_start_v1',
             'difficulty_status': 'nominal_project_tier_not_empirically_ranked',
             'runtime_scope': 'project_RGBD_single_goal_not_official_multimodal_Stretch',
             'termination': strict_json_loads(
                 (folder / f'episode-{index}/harness/termination.json').read_text())}
    if not ready:
        score['reason'] = 'nonfinite_runtime_metrics'
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--model-port", type=int, default=18182)
    parser.add_argument("--image", default="strive-hm3d:local")
    parser.add_argument("--language-task-file", type=Path)
    parser.add_argument("--language-task-index", type=int)
    parser.add_argument("--max-steps-per-task", type=int, default=500)
    parser.add_argument("--max-model-calls", type=int, default=60)
    parser.add_argument("policy_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    language = args.language_task_file is not None
    if language != (args.language_task_index is not None) or (language and args.policy_args):
        parser.error('Language mode requires both task file and index, without continuous policy arguments')
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    root = Path(os.environ["STRIVE_ROOT"]).resolve(strict=True)
    data = args.data_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    policy_args = args.policy_args[1:] if args.policy_args[:1] == ["--"] else args.policy_args
    settings = {
        "STRIVE_LLM_CLIENT": "self_hosted", "LLM_PROVIDER": "self_hosted",
        "VLN_LVLM_BASE_URL": f"http://127.0.0.1:{args.model_port}/v1",
        "VLN_LVLM_MODEL": "huaweivln-qwen3.5-4b",
        "VLN_LVLM_API_KEY": os.environ["VLN_LVLM_API_KEY"], "VLN_LVLM_GUIDED_JSON": "1",
        "VLN_LVLM_TIMEOUT_S": "180", "VLN_LVLM_PARSE_RETRIES": "1", "STRIVE_VLM_WORKERS": "1",
        "VLN_LVLM_TRANSPORT_RETRIES": "0",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
        "HF_HOME": "/root/.cache/huggingface", "SAM_CHECKPOINT": "/weights/sam_vit_h_4b8939.pth",
        "GROUNDING_DINO_PATH": "/opt/mmdetection",
        "GROUNDING_DINO_CONFIG": "/opt/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py",
        "GROUNDING_DINO_CHECKPOINT": "/weights/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth",
        "LLM_OFFLINE": "0", "STRIVE_LLM_FALLBACK": "0", "STRIVE_EVIDENCE_REUSE": "0",
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "MAGNUM_LOG": "quiet", "HABITAT_SIM_LOG": "quiet",
    }
    prefix = f"strive-goat-language-{args.language_task_index:03d}-" if language else "strive-goat-sequence-"
    run_id = prefix + datetime.now().strftime("%Y%m%d-%H%M%S")
    policy = ['scripts/run_goat.py', *policy_args, '--output', str(output / 'episode')]
    if language:
        policy, expected_identity, instruction = language_inputs(
            args.language_task_file.resolve(strict=True), args.language_task_index,
            data, output, args.max_steps_per_task, args.max_model_calls)
        policy += ['--run_id', run_id]
        settings['HM3D_DATA_PATH'] = str(data)
        settings['MP3D_DATA_PATH'] = str(data)
        (output / 'navigation').mkdir()
    command = ["docker", "run", "--rm", "--name", run_id, "--gpus", f"device={args.gpu}",
               "--network", "host", "--shm-size=12g", "--entrypoint", "/bin/bash"]
    for name in settings:
        command += ["-e", name]
    for mount in (
        f"{root}:/workspace/STRIVE:ro", f"{data}:{data}:ro",
        f"{output}:{output}", f"{os.environ['HF_HOME_HOST']}:/root/.cache/huggingface:ro",
        f"{os.environ['SAM_CHECKPOINT']}:{settings['SAM_CHECKPOINT']}:ro",
        f"{os.environ['GROUNDING_DINO_CHECKPOINT']}:{settings['GROUNDING_DINO_CHECKPOINT']}:ro",
    ):
        command += ["-v", mount]
    if language:
        command += ['-v', f'{output}/navigation:/workspace/STRIVE/logs']
    command += ["-w", "/workspace/STRIVE", image_id, "-lc", 'exec "$@"', "bash", "python", *policy]
    receipt = {
        "run_id": run_id, "command": command, "observation_mode": "multi", "decision_owner": "agent",
        "environment": {k: v for k, v in settings.items() if "KEY" not in k},
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in root.rglob("*.py") if "__pycache__" not in p.parts},
        "image_id": image_id,
    }
    if language:
        receipt['language_task'] = {
            'index': args.language_task_index, 'identity': expected_identity,
            'source_sha256': hashlib.sha256(args.language_task_file.read_bytes()).hexdigest(),
            'protocol': 'goat_language_single_parent_start_v1',
        }
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "policy.log").open("x") as stream:
        process = subprocess.run(command, env=dict(os.environ, **settings), stdout=stream,
                                 stderr=subprocess.STDOUT)
    if language and process.returncode == 0:
        try:
            episode_dir = output / f'navigation/task/episode-{args.language_task_index}'
            scoring_command = [
                'docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python',
                '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'MAGNUM_LOG=quiet',
                '-e', 'HABITAT_SIM_LOG=quiet',
                '-v', f'{root}:/workspace/STRIVE:ro', '-v', f'{output}:{output}:ro',
                '-w', '/workspace/STRIVE', image_id,
                '-m', 'benchmark.habitat_evaluation', str(episode_dir),
            ]
            independent = strict_json_loads(subprocess.check_output(scoring_command, text=True))
            score = language_score(output, args.language_task_index, expected_identity, instruction,
                                   independent)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            score = {'aggregate_ready': False, 'metrics': None,
                     'unavailable_metrics': {'navigation': f'{type(error).__name__}: {error}'}}
        (output / 'evaluation.json').write_text(json.dumps(score, allow_nan=False, indent=2) + '\n')
    (output / "process_result.json").write_text(json.dumps({
        "run_id": run_id, "exit_code": process.returncode,
        "status": "execution_completed" if process.returncode == 0 else "execution_error",
    }, indent=2) + "\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
