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
        overlap_scores = []
        overlap_scores1 = []
        overlap_scores2 = []
        eval_pcd = entity.pcd
        for ref_entity in ref_entities:
            if eval_pcd.point.positions.shape[0] == 0:
                break
            if ref_entity.alive == False:
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
            # nonoverlap_condition = overlap_condition.logical_not()
            # eval_pcd = eval_pcd.select_by_index(
            #     o3d.core.Tensor(nonoverlap_condition.cpu().numpy(),
            #                     device=self.pcd_device).nonzero()[0])
            overlap_scores1.append(
                (overlap_condition1.sum() / (overlap_condition1.shape[0] + 1e-6)).cpu().numpy())
            overlap_scores2.append(
                (overlap_condition2.sum() / (overlap_condition2.shape[0] + 1e-6)).cpu().numpy())
            overlap_scores.append(
                (overlap_condition_all.sum() / (overlap_condition_all.shape[0] + 1e-6)).cpu().numpy())
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
                        source_tag = entity.tag if entity.tag in entity.num_list else target
                        argmax_entity.tag = target
                        argmax_entity.pcd = eval_pcd
                        argmax_entity.position = entity.position
                        # 同一实例关联成功后同步图像与其 bbox，供证据版本比较。
                        argmax_entity.rgb, argmax_entity.bbox = entity.rgb, entity.bbox
                        argmax_entity.confidence = entity.confidence
                        argmax_entity.num_list[target] = entity.num_list.get(source_tag, 1)
                        argmax_entity.conf_list[target] = entity.conf_list.get(
                            source_tag,
                            [entity.confidence],
                        )
                    else:
                        entity.pcd = eval_pcd
                        ref_entities.append(entity)
                        entity_indices.append(len(ref_entities) - 1)
            else:
                argmax_entity = ref_entities[arg_overlap_index]
                argmax_entity.rgb, argmax_entity.bbox = entity.rgb, entity.bbox
                argmax_entity.pcd = gpu_merge_pointcloud(argmax_entity.pcd, eval_pcd)
                argmax_entity.position = np.mean(argmax_entity.pcd.point.positions.cpu().numpy(),
                                                 axis=0)
                for node in entity.nodes:
                    if node not in argmax_entity.nodes:
                        argmax_entity.nodes.append(node)

                alpha = 2
                ori_tag = argmax_entity.tag
                new_tag = entity.tag

                if new_tag not in argmax_entity.num_list.keys():
                    argmax_entity.num_list[new_tag] = entity.num_list[new_tag]
                    argmax_entity.conf_list[new_tag] = entity.conf_list[new_tag]
                else:
                    n1 = argmax_entity.num_list[new_tag]
                    n2 = entity.num_list[new_tag]
                    c1 = argmax_entity.conf_list[new_tag]
                    c2 = entity.conf_list[new_tag]
                    w1 = (c1**alpha * n1) / (c1**alpha * n1 + c2**alpha * n2)
                    w2 = 1 - w1
                    argmax_entity.conf_list[new_tag] = c1 * w1 + c2 * w2
                    argmax_entity.num_list[new_tag] = n1 + n2

                if new_tag == ori_tag:
                    argmax_entity.tag = new_tag
                    argmax_entity.confidence = argmax_entity.conf_list[new_tag]
                else:
                    # 部分 verifier 路径会临时把 tag 置为 nothing
                    # 以退出当前候选，但历史类别字典可能来自旧版本日志或其它
                    # 路径而缺 key。这里补齐默认值，避免地图关联阶段崩溃。
                    if ori_tag not in argmax_entity.num_list:
                        argmax_entity.num_list[ori_tag] = 1
                    if ori_tag not in argmax_entity.conf_list:
                        argmax_entity.conf_list[ori_tag] = argmax_entity.confidence
                    n1 = argmax_entity.num_list[ori_tag]
                    n2 = argmax_entity.num_list[new_tag]
                    c1 = argmax_entity.conf_list[ori_tag]
                    c2 = argmax_entity.conf_list[new_tag]
                    w1 = (c1**alpha * n1) / (c1**alpha * n1 + c2**alpha * n2)
                    w2 = 1 - w1
                    if w1 > w2:
                        argmax_entity.tag = ori_tag
                        argmax_entity.confidence = c1
                    else:
                        argmax_entity.tag = new_tag
                        argmax_entity.confidence = c2


                ref_entities[arg_overlap_index] = argmax_entity
                entity_indices.append(arg_overlap_index)
    entity_indices = [int(i) for i in entity_indices]
    return ref_entities, entity_indices
