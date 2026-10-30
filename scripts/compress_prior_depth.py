#!/usr/bin/env python3
"""Export a complete prior map using lossless compressed depth evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.depth_storage import compress_bundle_depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    receipt = compress_bundle_depth(args.source, args.output)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('files', 'source_index_sha256')}))
