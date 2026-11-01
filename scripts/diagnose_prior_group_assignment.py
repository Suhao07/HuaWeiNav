#!/usr/bin/env python3
"""Test weak-observation assignment to frozen cross-position support groups.

This writes a counterfactual trace, never a map. No GT is used in decisions.
Reference groups stay fixed so newly assigned observations cannot self-reinforce.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mapping.object_fusion import score_object_candidates
from prior_map.bundle import load_bundle
from prior_map.observations import load_fusion_observation, write_json


def main():
    """Preserve original 0.25 admission and experimental 0.10 score margin."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('trace',type=Path)
    parser.add_argument('groups',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    records={o['observation_id']:o for o in load_bundle(args.bundle)['observations']}
    receipt=json.loads((args.bundle/'build_receipt.json').read_text())
    trace=json.loads(args.trace.read_text())
    groups=json.loads(args.groups.read_text())
    references=defaultdict(list)
    for index,group in enumerate(groups['components']):
        if group['camera_position_count']>=2:
            references[group['uid']].append((f"{group['uid']}:support:{index}",group['observations']))
    references={uid:items for uid,items in references.items() if len(items)>=2}
    members={oid:uid for items in references.values() for uid,ids in items for oid in ids}

    @lru_cache(maxsize=64)
    def load(oid):
        return load_fusion_observation(args.bundle,records[oid],voxel_m=receipt['voxel_m'],
            max_depth_m=receipt['max_depth_m'],profile=receipt['capture_manifest'].get('capture_profile'))[0]

    observations,decisions=[],[]
    for original in trace['observations']:
        oid,uid=original['observation_id'],original['assigned_uid']
        if oid in members:
            uid=members[oid]
        elif uid in references:
            candidates=[]
            for candidate,ids in references[uid]:
                scores=[]
                for member in ids:
                    pooled,a,b=score_object_candidates(load(oid),[load(member)])
                    if pooled[0]>=.25:
                        scores.append((float((a[0]+b[0])/2),member))
                if scores:
                    score,witness=max(scores)
                    candidates.append(dict(uid=candidate,score=score,witness=witness))
            candidates.sort(key=lambda r:-r['score'])
            uid=(candidates[0]['uid'] if candidates and
                 (len(candidates)==1 or candidates[0]['score']-candidates[1]['score']>=.10) else None)
            decisions.append(dict(observation_id=oid,assigned_uid=uid,candidates=candidates))
        observations.append(dict(original,assigned_uid=uid))
    sources=[args.trace,args.groups,args.bundle/'observations.jsonl',args.bundle/'build_receipt.json',
             Path(__file__),Path('mapping/object_fusion.py'),Path('prior_map/observations.py')]
    write_json(args.output,dict(scope='counterfactual_weak_assignment_not_applied',
        selected_uids=list(references),observations=observations,decisions=decisions,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    print(json.dumps(dict(selected_instances=len(references),weak_observations=len(decisions),
                         assigned=sum(r['assigned_uid'] is not None for r in decisions))))


if __name__=='__main__':
    main()
