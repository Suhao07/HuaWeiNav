"""Execute one frozen prior-map comparison arm with exact task provenance."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def check_inputs(root, manifest_path, manifest):
    """Require the same code, task subset and map core for every comparison arm."""
    sources = json.loads((manifest_path.parent/'source_hashes.json').read_text())
    checks = [(root/name, digest) for name,digest in sources.items()]
    checks += [(manifest_path.parent/'tasks.json.gz', manifest['dataset_sha256']),
               (root/manifest['map_path']/'prior_map.json', manifest['map_core_sha256'])]
    if "layout_path" in manifest:
        checks.append((root/manifest["layout_path"], manifest["layout_sha256"]))
    if "observations_sha256" in manifest:
        checks.append((root/manifest["map_path"]/"observations.jsonl", manifest["observations_sha256"]))
    for path, expected in checks:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {path}')


def main():
    """Run the existing benchmark launcher; retain failures and verify episode identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--arm', choices=['none','objects','objects_evidence'], required=True)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    check_inputs(root,manifest_path,manifest)
    if args.check_only:
        print('Frozen code, dataset and map hashes match')
        return
    if args.key_file is None:
        parser.error('--key-file is required to run the configured model')
    output = root/manifest['run_output_directories'][args.arm]
    if output.exists():
        raise FileExistsError(f'Run output already exists: {output}')
    receipt_dir = manifest_path.parent/f'execution-{args.arm}'
    receipt_dir.mkdir(exist_ok=False)
    key = next(line.strip() for line in args.key_file.read_text().splitlines() if line.strip().startswith('ark-'))
    env = dict(os.environ,ARK_API_KEY=key,LLM_MODEL=manifest['model'],IMAGE_TAG=manifest['image'],
        LLM_PROVIDER='ark',LLM_OFFLINE='0',STRIVE_LLM_FALLBACK='0',STRIVE_LLM_PARSE_FALLBACK='0',
        STRIVE_EVIDENCE_REUSE='0',PYTHONHASHSEED='0',CONTAINER_NAME=f'strive-prior-{args.arm}')
    command = manifest['commands'][args.arm]
    receipt = dict(status='running',arm=args.arm,command=command,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_hashes_sha256=hashlib.sha256((manifest_path.parent/'source_hashes.json').read_bytes()).hexdigest())
    path = receipt_dir/'receipt.json'
    path.write_text(json.dumps(receipt,indent=2))
    started=time.monotonic()
    with (receipt_dir/'run.log').open('w') as stream:
        process=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
    receipt.update(status='failed' if process.returncode else 'verifying',returncode=process.returncode,
                   elapsed_s=time.monotonic()-started)
    path.write_text(json.dumps(receipt,indent=2))
    process.check_returncode()
    first = int(command[command.index('--start_episode')+1]) if '--start_episode' in command else 0
    expected_episodes = manifest['runtime_episodes'][first:]
    for index, expected in enumerate(expected_episodes, start=first):
        actual=json.loads((output/f'episode-{index}'/'episode_identity.json').read_text())
        if actual != expected:
            receipt.update(status='identity_mismatch',episode_index=index)
            path.write_text(json.dumps(receipt,indent=2))
            raise ValueError(f'Frozen episode identity differs at index {index}')
    with (output/'metrics.csv').open() as stream:
        metrics=list(csv.DictReader(stream))
    if len(metrics)!=len(expected_episodes):
        raise ValueError('Metric count differs from frozen task count')
    check_inputs(root,manifest_path,manifest)
    receipt.update(status='complete',metrics=metrics)
    path.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(arm=args.arm,status='complete',episodes=len(metrics))))


if __name__=='__main__':
    main()
