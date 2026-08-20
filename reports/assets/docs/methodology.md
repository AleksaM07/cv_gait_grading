# Methodology

The primary task is to estimate the simulator-derived quality of a MuJoCo
walking policy using only external side-view RGB video at inference time.
Policy identity, checkpoint names, rewards, commands, contacts, and simulator
state are used only for rendering, label construction, grouping, and analysis.

The active transfer pipeline samples two deterministic 16-frame windows from
each clip. A frozen torchvision R3D-18 pretrained on Kinetics-400 produces a
512-dimensional embedding. PCA/whitening with 64 components is fitted only on
975 unlabeled walking clips (580 CMU renders and 395 GaHu real videos), never on
the labeled validation or test targets.

The supervised dataset contains 54 side-view MuJoCo clips: six standardized
scenarios from each of nine policy/checkpoint groups. The fixed split is
36/6/12 clips across six/one/two policy groups. Nested leave-one-policy-out
evaluation rotates every policy through the outer test fold, while Ridge alpha
selection happens only inside the remaining groups.

Single-video inference returns a numeric research score plus camera and
embedding-domain checks. Policy-level inference loads the model once, scores
six scenarios, and reports their mean, standard deviation, minimum, maximum,
and per-scenario details. The aggregate is invalid if an expected scenario is
missing or rejected as OOD.

The classical dense-flow/PCA model remains an interpretability and comparison
baseline. See `scoring_definition.md` for exact target semantics.
