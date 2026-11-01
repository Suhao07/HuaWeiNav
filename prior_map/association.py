"""Observation-owned prior-map association with bounded deferred resolution."""
from copy import deepcopy
from itertools import combinations
from dataclasses import asdict, dataclass

from mapping.object_fusion import clouds_may_overlap, merge_object_observation, score_object_candidates


@dataclass(frozen=True)
class AssociationConfig:
    """Explicit experimental admission thresholds; units are coverage fractions."""
    overlap: float = .25
    margin: float = .10
    coview_iou: float = .05
    resolution_support: float = .50
    balanced_coverage: bool = False


class PriorAssociation:
    """Keep raw evidence separate from mutable instances.

    Args:
        load_observation: Callable returning a fresh (ObjectNode, mask) by ID.
            It reconstructs the original filtered cloud, never an instance cloud.
        config: Frozen experimental admission thresholds.

    Instance P_j = Fuse(P_i for assignment[i] == j), in observation order.
    A deferred observation contributes neither geometry nor category statistics.
    """

    def __init__(self, load_observation, config=AssociationConfig()):
        self.load = load_observation
        self.config = config
        self.entities, self.members, self.versions = {}, {}, {}
        self.assignment, self.pending, self.frames = {}, {}, {}
        self.events, self.dirty = [], set()
        self.coview_conflicts = []
        self.order = {}

    def _scores(self, entity, references):
        pooled, forward, reverse = score_object_candidates(entity, list(references.values()))
        admitted = [score >= self.config.overlap for score in pooled]
        if self.config.balanced_coverage:
            # 双向等权：s=(coverage(P,Q)+coverage(Q,P))/2，消除点数加权。
            pooled = [(a + b) / 2 for a, b in zip(forward, reverse)]
        return sorted((dict(uid=uid, overlap=float(s), new_to_old=float(a), old_to_new=float(b))
                       for uid, s, a, b, keep in zip(references, pooled, forward, reverse, admitted)
                       if keep), key=lambda row: -row['overlap'])

    def _choose(self, oid, entity, mask, references, members):
        scores = self._scores(entity, references)
        if not scores:
            return entity.uid, 'new_instance', scores
        best = scores[0]
        if len(scores) > 1 and best['overlap'] - scores[1]['overlap'] < self.config.margin:
            return None, 'candidate_margin', scores
        correction = self.pending.get(oid, {}).get('correction')
        if correction is not None and best['uid'] == correction['uid']:
            support = 0.
            for member in members[best['uid']]:
                if self.order[member] > correction['after_order'] and self.frames[member] != self.frames[oid]:
                    raw, _ = self.load(member)
                    values, _, _ = score_object_candidates(entity, [raw])
                    support = max(support, float(values[0]))
            if support < self.config.resolution_support:
                return None, 'await_post_correction_evidence', scores
        conflict = False
        independent_support = 0.
        for member in members[best['uid']]:
            old, old_mask = self.load(member)
            if self.frames[member] == self.frames[oid]:
                iou = (mask & old_mask).sum() / (mask | old_mask).sum()
                conflict |= iou < self.config.coview_iou
            else:
                support, _, _ = score_object_candidates(entity, [old])
                independent_support = max(independent_support, float(support[0]))
        # 同帧冲突需要另一帧的原始观察支持；实例累计点云不能自证身份。
        if conflict and independent_support < self.config.resolution_support:
            return None, 'coview_conflict', scores
        return best['uid'], 'unique_candidate', scores

    def _commit(self, oid, uid, reason):
        entity, _ = self.load(oid)
        if uid in self.entities:
            merge_object_observation(self.entities[uid], entity)
            self.members[uid].append(oid)
        else:
            entity.uid = uid
            self.entities[uid] = entity
            self.members[uid] = [oid]
        self.versions[uid] = self.versions.get(uid, 0) + 1
        self.assignment[oid] = uid
        self.pending.pop(oid, None)
        self.dirty.add(uid)
        self.events.append(dict(observation_id=oid, assigned_uid=uid, reason=reason))

    def _defer(self, oid, reason, scores, checked_versions):
        correction = self.pending.get(oid, {}).get('correction')
        self.assignment[oid] = None
        self.pending[oid] = dict(reason=reason, candidates=[s['uid'] for s in scores],
                                 checked_versions=dict(checked_versions))
        if correction is not None:
            self.pending[oid]['correction'] = correction
        self.events.append(dict(observation_id=oid, assigned_uid=None, reason=reason))

    def _record_coview_conflicts(self, decisions, stage):
        """Record conflicting proposed memberships without changing admission.

        IoU(M_i,M_j) < tau is a conflict cue, not an instance-identity verdict.
        Include same-frame committed members when deferred observations resolve.
        """
        groups = {}
        for oid, uid, _, _ in decisions:
            if uid is None:
                continue
            key = (self.frames[oid], uid)
            if key not in groups:
                groups[key] = {member for member in self.members.get(uid, [])
                               if self.frames[member] == self.frames[oid]}
            groups[key].add(oid)
        for (frame, uid), ids in groups.items():
            masks = {oid: self.load(oid)[1] for oid in ids}
            for first, second in combinations(sorted(ids, key=self.order.__getitem__), 2):
                a, b = masks[first], masks[second]
                intersection = int((a & b).sum())
                iou = float(intersection / (a | b).sum())
                if iou < self.config.coview_iou:
                    self.coview_conflicts.append(dict(frame_id=frame, uid=uid,
                        first=first, second=second, mask_iou=iou, stage=stage,
                        first_covered=float(intersection/a.sum()),
                        second_covered=float(intersection/b.sum())))

    def add_frame(self, frame_id, observation_ids):
        """Score a frame against fixed instances and immutable new-instance seeds.

        Commit after all choices are made. Raw seeds support within-frame duplicate
        matching while avoiding score changes from an already merged observation.
        """
        # 记录评分时的版本；提交后的版本可能包含尚未评估的新证据。
        checked_versions = dict(self.versions)
        references = dict(self.entities)
        members = {uid: list(ids) for uid, ids in self.members.items()}
        decisions = []
        for oid in observation_ids:
            if oid in self.frames:
                raise ValueError(f'Duplicate observation: {oid}')
            self.frames[oid] = frame_id
            self.order[oid] = len(self.order)
            entity, mask = self.load(oid)
            uid, reason, scores = self._choose(oid, entity, mask, references, members)
            decisions.append((oid, uid, reason, scores))
            if uid is not None and uid not in references:
                references[uid], members[uid] = entity, [oid]
        self._record_coview_conflicts(decisions, 'frame_commit')
        for oid, uid, reason, scores in decisions:
            if uid is None:
                self._defer(oid, reason, scores, checked_versions)
            else:
                self._commit(oid, uid, reason)
        self.resolve_pending()

    def resolve_pending(self):
        """One event-driven pass; commits schedule related work for the next pass."""
        changed, self.dirty = self.dirty, set()
        checked_versions = dict(self.versions)
        decisions = []
        for oid, record in list(self.pending.items()):
            entity, mask = self.load(oid)
            relevant = [uid for uid in changed
                        if self.versions.get(uid) != record['checked_versions'].get(uid)
                        and (uid in record['candidates'] or
                             (uid in self.entities and clouds_may_overlap(entity.pcd, self.entities[uid].pcd)))]
            if not relevant:
                continue
            uid, reason, scores = self._choose(oid, entity, mask, self.entities, self.members)
            # 撤销后失去候选仍保留待定；新实例由后续独立观察建立。
            if reason == 'new_instance':
                uid, reason = None, 'await_independent_observation'
            decisions.append((oid, uid, reason, scores))
        self._record_coview_conflicts(decisions, 'pending_resolution')
        for oid, uid, reason, scores in decisions:
            if uid is None:
                self._defer(oid, reason, scores, checked_versions)
            else:
                self._commit(oid, uid, 'resolved_' + reason)

    def retract(self, oid, *, reason):
        """Apply an explicit evidence-backed correction and rebuild only its instance.

        The caller must supply the conflict reason. This operation offers local
        correction; it does not infer ground-truth identity or split the whole map.
        """
        uid = self.assignment[oid]
        if uid is None:
            raise ValueError(f'Observation is already pending: {oid}')
        remaining = sorted((i for i in self.members[uid] if i != oid), key=self.order.__getitem__)
        if remaining:
            rebuilt, _ = self.load(remaining[0])
            rebuilt.uid = uid
            for member in remaining[1:]:
                entity, _ = self.load(member)
                merge_object_observation(rebuilt, entity)
            self.entities[uid], self.members[uid] = rebuilt, remaining
        else:
            del self.entities[uid], self.members[uid]
        self.versions[uid] += 1
        self._defer(oid, reason, [], self.versions)
        # Include the removed candidate so its correction is visible on next pass.
        self.pending[oid]['candidates'] = [uid]
        self.pending[oid]['correction'] = dict(uid=uid, after_order=max(self.order.values()))
        self.dirty.add(uid)

    def trace(self):
        """Return final assignments plus transitions and unresolved evidence."""
        return dict(config=asdict(self.config), observations=[
            dict(observation_id=oid, frame_id=self.frames[oid], assigned_uid=uid)
            for oid, uid in self.assignment.items()], pending=deepcopy(self.pending), events=list(self.events),
            coview_conflicts=deepcopy(self.coview_conflicts))
