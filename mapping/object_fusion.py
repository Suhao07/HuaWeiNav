"""Shared instance fusion for online mapping and task-free capture replay."""
import numpy as np
import torch
from mapping_utils.geometry import gpu_merge_pointcloud
from scipy.spatial import cKDTree


def nearest_distances(source, reference):
    """Return min_q ||p-q|| for each source point using an exact spatial index.

    Fusion previously allocated an N x M dense distance matrix for this query.
    cKDTree queries exact Euclidean neighbors (eps=0); point coordinates and
    overlap thresholds remain unchanged. The returned tensor keeps the source
    coordinate dtype and the CPU interface used by the association policy.
    """
    points = source.point.positions.cpu().numpy()
    other = reference.point.positions.cpu().numpy()
    distances, _ = cKDTree(other).query(points, eps=0)
    return torch.from_numpy(distances.astype(points.dtype, copy=False))


def clouds_may_overlap(first, second):
    """Conservative AABB bound for the strict 0.1 m overlap test.

    d_box = ||max(lo_A-hi_B, lo_B-hi_A, 0)|| is a lower bound on every
    point-pair distance. d_box >= 0.1 implies exactly zero qualifying pairs.
    """
    a = first.point.positions.cpu().numpy()
    b = second.point.positions.cpu().numpy()
    gap = np.maximum(np.maximum(a.min(axis=0) - b.max(axis=0),
                                 b.min(axis=0) - a.max(axis=0)), 0)
    return np.linalg.norm(gap) < .1


def score_object_candidates(entity, ref_entities):
    """Read-only pooled and directional surface coverage in reference order.

    s=(sum(d(P,Q)<0.1)+sum(d(Q,P)<0.1))/(|P|+|Q|+1e-6).
    Inactive entries keep their index with score -inf. Input clouds are nonempty.
    """
    overlap_scores, overlap_scores1, overlap_scores2 = [], [], []
    eval_pcd = entity.pcd
    for ref_entity in ref_entities:
        if not ref_entity.alive:
            overlap_scores.append(float("-inf"))
            overlap_scores1.append(0.)
            overlap_scores2.append(0.)
            continue
        if not clouds_may_overlap(eval_pcd, ref_entity.pcd):
            overlap_scores.append(0.)
            overlap_scores1.append(0.)
            overlap_scores2.append(0.)
            continue
        cdist1 = nearest_distances(eval_pcd, ref_entity.pcd)
        cdist2 = nearest_distances(ref_entity.pcd, eval_pcd)
        cdist_all = torch.cat([cdist1, cdist2], dim=0)
        overlap_condition1 = (cdist1 < 0.1)
        overlap_condition2 = (cdist2 < 0.1)
        overlap_condition_all = (cdist_all < 0.1)
        overlap_scores1.append(
            (overlap_condition1.sum() / (overlap_condition1.shape[0] + 1e-6)).cpu().numpy())
        overlap_scores2.append(
            (overlap_condition2.sum() / (overlap_condition2.shape[0] + 1e-6)).cpu().numpy())
        overlap_scores.append(
            (overlap_condition_all.sum() / (overlap_condition_all.shape[0] + 1e-6)).cpu().numpy())
    return overlap_scores, overlap_scores1, overlap_scores2


def merge_object_observation(argmax_entity, entity):
    """Commit one admitted observation using the existing STRIVE aggregation.

    Mutates the destination cloud, category statistics and representative image.
    Callers own identity admission; raw observations must remain immutable.
    """
    argmax_entity.rgb, argmax_entity.bbox = entity.rgb, entity.bbox
    argmax_entity.pcd = gpu_merge_pointcloud(argmax_entity.pcd, entity.pcd)
    argmax_entity.position = np.mean(argmax_entity.pcd.point.positions.cpu().numpy(),
                                     axis=0)
    for node in entity.nodes:
        if node not in argmax_entity.nodes:
            argmax_entity.nodes.append(node)
    accumulate_label_vote(argmax_entity, entity.tag, entity.num_list[entity.tag], entity.conf_list[entity.tag])


def accumulate_label_vote(argmax_entity, new_tag, n2, c2):
    """按票重 w=c²n 加权更新同类置信度，再比较类别票重。"""
    alpha = 2
    ori_tag = argmax_entity.tag
    if c2**alpha * n2 == 0:
        return

    if (new_tag not in argmax_entity.num_list
            or argmax_entity.conf_list[new_tag]**alpha * argmax_entity.num_list[new_tag] == 0):
        argmax_entity.num_list[new_tag] = n2
        argmax_entity.conf_list[new_tag] = c2
    else:
        n1 = argmax_entity.num_list[new_tag]
        c1 = argmax_entity.conf_list[new_tag]
        w1 = (c1**alpha * n1) / (c1**alpha * n1 + c2**alpha * n2)
        w2 = 1 - w1
        argmax_entity.conf_list[new_tag] = c1 * w1 + c2 * w2
        argmax_entity.num_list[new_tag] = n1 + n2

    if new_tag == ori_tag:
        argmax_entity.tag = new_tag
        argmax_entity.confidence = argmax_entity.conf_list[new_tag]
    else:
        n1 = argmax_entity.num_list[ori_tag]
        n2 = argmax_entity.num_list[new_tag]
        c1 = argmax_entity.conf_list[ori_tag]
        c2 = argmax_entity.conf_list[new_tag]
        if c1**alpha * n1 > c2**alpha * n2:
            argmax_entity.tag = ori_tag
            argmax_entity.confidence = c1
        else:
            argmax_entity.tag = new_tag
            argmax_entity.confidence = c2


def label_probabilities(entity):
    """Category posterior from the fusion's own votes: p(l)=c_l^2 n_l / sum_k c_k^2 n_k."""
    votes = {tag: float(entity.conf_list[tag]) ** 2 * float(n) for tag, n in entity.num_list.items()}
    total = sum(votes.values())
    return {tag: vote / total for tag, vote in votes.items() if vote > 0} if total > 0 else {}

def merge_duplicate_instances(entities, voxel_m):
    """Merge instances describing one surface; observation-to-instance fusion never re-merges.

    S=voxel(P, voxel_m) removes point multiplicity from repeated observations;
    c(X,Y)=mean_x 1[d(x,Y)<0.1]. A and B merge iff c(S_A,S_B)>0.85 and
    c(S_B,S_A)>0.85, the same-object test of the target branch. One-sided
    containment (part on whole, small object on large) does not merge. Every
    label vote of the merged instance carries over, not only its current tag.

    Returns:
        Surviving entities and {merged_uid: surviving_uid}.
    """
    entities = list(entities)
    surfaces = [e.pcd.voxel_down_sample(voxel_m) for e in entities]
    merged = {}

    def same_surface(a, b):
        return (clouds_may_overlap(a, b) and float((nearest_distances(a, b) < .1).float().mean()) > .85
                and float((nearest_distances(b, a) < .1).float().mean()) > .85)

    changed = True
    while changed:
        changed = False
        for i in range(len(entities)):
            j = i + 1
            while j < len(entities):
                if same_surface(surfaces[i], surfaces[j]):
                    merge_object_observation(entities[i], entities[j])
                    for tag, n in entities[j].num_list.items():
                        if tag != entities[j].tag:
                            accumulate_label_vote(entities[i], tag, n, entities[j].conf_list[tag])
                    merged[entities[j].uid] = entities[i].uid
                    surfaces[i] = entities[i].pcd.voxel_down_sample(voxel_m)
                    del entities[j], surfaces[j]
                    changed = True
                else:
                    j += 1
    for uid in merged:
        while merged[uid] in merged:
            merged[uid] = merged[merged[uid]]
    return entities, merged


def associate_object_entities(ref_entities, eval_entities, *, target, is_target_tag):
    """Associate observed clouds using the existing STRIVE overlap policy.

    The caller supplies task semantics explicitly. Offline mapping passes an
    empty target and a predicate returning False, preserving the background
    fusion path. Overlap is the bidirectional fraction of nearest distances
    below 0.1 m; a score of 0.25 admits fusion.
    """
    entity_indices = []
    for entity in eval_entities:
        if len(ref_entities) == 0:
            ref_entities.append(entity)
            entity_indices.append(0)
            continue
        eval_pcd = entity.pcd
        overlap_scores, overlap_scores1, overlap_scores2 = score_object_candidates(entity, ref_entities)
        max_overlap_score = np.max(overlap_scores)
        arg_overlap_index = np.argmax(overlap_scores)


        if max_overlap_score < 0.25:
            entity.pcd = eval_pcd
            ref_entities.append(entity)
            entity_indices.append(len(ref_entities) - 1)
        else:
            # target class
            if is_target_tag(entity.tag):
                argmax_entity = ref_entities[arg_overlap_index]
                print(argmax_entity.tag)
                if argmax_entity.tag == 'nothing':
                    continue
                else:
                    overlap_score = overlap_scores[arg_overlap_index]
                    overlap_score1 = overlap_scores1[arg_overlap_index]
                    overlap_score2 = overlap_scores2[arg_overlap_index]
                    # check if this 2 is the same object:
                    if overlap_score > 0.85 and overlap_score1 > 0.85 and overlap_score2 > 0.85:
                        argmax_entity.num_list.pop(argmax_entity.tag)
                        argmax_entity.conf_list.pop(argmax_entity.tag)
                        argmax_entity.tag = target
                        argmax_entity.pcd = eval_pcd
                        argmax_entity.position = entity.position
                        # 同一实例关联成功后同步图像与其 bbox，供证据版本比较。
                        argmax_entity.rgb, argmax_entity.bbox = entity.rgb, entity.bbox
                        argmax_entity.confidence = entity.confidence
                        argmax_entity.num_list[target] = entity.num_list[entity.tag]
                        argmax_entity.conf_list[target] = entity.conf_list[entity.tag]
                    else:
                        entity.pcd = eval_pcd
                        ref_entities.append(entity)
                        entity_indices.append(len(ref_entities) - 1)
            else:
                argmax_entity = ref_entities[arg_overlap_index]
                merge_object_observation(argmax_entity, entity)
                ref_entities[arg_overlap_index] = argmax_entity
                entity_indices.append(arg_overlap_index)
    entity_indices = [int(i) for i in entity_indices]
    return ref_entities, entity_indices
