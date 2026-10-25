"""Legacy import facade; domain modules are loaded only when requested.

This avoids importing perception, verification and model clients for a contract.
The facade is removed when its external callers have migrated.
"""
from importlib import import_module

_EXPORTS = {
    'compile_instruction_plan': ('compiler', 'compile_instruction_plan'),
    'ConstraintEvaluation': ('constraints', 'ConstraintEvaluation'),
    'ConstraintEvaluator': ('constraints', 'ConstraintEvaluator'),
    'AnchorSearchLedger': ('concept_matcher', 'AnchorSearchLedger'),
    'ConceptMatchRecord': ('concept_matcher', 'ConceptMatchRecord'),
    'RuntimeConceptMatcher': ('concept_matcher', 'RuntimeConceptMatcher'),
    'ConceptQuery': ('contracts', 'ConceptQuery'),
    'Constraint': ('contracts', 'Constraint'),
    'ExecutionPolicy': ('contracts', 'ExecutionPolicy'),
    'InstructionPlan': ('contracts', 'InstructionPlan'),
    'SearchPriors': ('contracts', 'SearchPriors'),
    'StriveInstructionSpec': ('contracts', 'StriveInstructionSpec'),
    'TargetQuery': ('contracts', 'TargetQuery'),
    'StriveInstructionParser': ('parser', 'StriveInstructionParser'),
    'extract_dataset_target': ('parser', 'extract_dataset_target'),
    'render_instruction_context': ('prompt_context', 'render_instruction_context'),
    'DynamicRelationService': ('relation_verifier', 'DynamicRelationService'),
    'VLMRelationVerifier': ('relation_verifier', 'VLMRelationVerifier'),
    'DynamicSemanticEdgeVerifier': ('semantic_edges', 'DynamicSemanticEdgeVerifier'),
    'RelationPairLedger': ('semantic_edges', 'RelationPairLedger'),
    'RelationQuery': ('semantic_edges', 'RelationQuery'),
    'SemanticEdge': ('semantic_edges', 'SemanticEdge'),
    'SemanticEdgeCache': ('semantic_edges', 'SemanticEdgeCache'),
    'InstructionSpatialGraph': ('spatial_graph', 'InstructionSpatialGraph'),
    'ObjectNodeRecord': ('spatial_graph', 'ObjectNodeRecord'),
    'ViewNode': ('spatial_graph', 'ViewNode'),
    'CandidateInstance': ('verifier', 'CandidateInstance'),
    'FinalInstructionVerifier': ('verifier', 'FinalInstructionVerifier'),
    'EvidenceStore': ('verifier', 'EvidenceStore'),
    'VerificationResult': ('verifier', 'VerificationResult'),
    'candidate_from_object': ('verifier', 'candidate_from_object'),
    'candidate_uid_from_object': ('verifier', 'candidate_uid_from_object'),
    'ViewAttempt': ('view_control', 'ViewAttempt'),
    'ViewControlState': ('view_control', 'ViewControlState'),
    'ViewpointProposal': ('view_control', 'ViewpointProposal'),
    'view_quality_from_evidence': ('view_control', 'view_quality_from_evidence'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """Resolve a legacy symbol without eager cross-domain imports."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), symbol)
    globals()[name] = value
    return value
