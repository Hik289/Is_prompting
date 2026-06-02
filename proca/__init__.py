"""ProCA: Prototype-driven Contrastive Cultural Alignment.

Implementation of the ProCA framework for adapting LLMs to diverse
cultural value systems via prototype-anchored contrastive learning.

Modules
-------
prototypes -- cultural prototype space (PCA on WVS responses)
synthesis -- theory-guided social interaction synthesis
encoders -- context / value / intent encoders
losses -- KL, contrastive, and unified objectives
model -- ProCAModel: encoders + losses + LoRA orchestration
refinement -- cultural relevance scoring + iterative filtering
train -- training loop (Algorithm 1)
eval.* -- WVS and cross-lingual evaluators + baselines
ablations.* -- dialogue_only, intent_only, reasoning_only, teacher_swap
"""

__version__ = "0.1.0"
