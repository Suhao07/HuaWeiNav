#!/usr/bin/env python3
"""Evaluate saved predictions using separate Habitat semantic annotations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.semantic_audit import audit_habitat_instances


def main():
    """Keep oracle diagnostics separate from the predicted map artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('--scene-dataset-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--semantic-cache')
    parser.add_argument('--scene', help='Local scene asset; checked against capture SHA256')
    args = parser.parse_args()
    result = audit_habitat_instances(args.bundle, args.scene_dataset_config, args.output, args.semantic_cache, args.scene)
    print(json.dumps({k: v for k, v in result['summary'].items()
                      if k not in ('instance_votes', 'label_pairs')}, indent=2))


if __name__ == '__main__':
    main()
