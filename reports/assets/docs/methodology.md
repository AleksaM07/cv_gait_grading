# Methodology

The primary method is intentionally classical and interpretable. It avoids using
policy identity, checkpoint names, XML identifiers, rewards, commands, contacts,
or simulator state during inference. Those fields are used only for labels,
validation, grouping, and analysis.

The leakage-safe model input is the RGB clip plus its FPS. The active experiment
uses 56 side-view clips, and grouped splitting keeps all clips from one
policy/checkpoint family in exactly one of train, validation, or test. The split
is 40/8/8 clips across five/one/one policy groups.

Only the overall source composite, survival-derived stability, and fall event
are trainable from the compact renderer export. Missing component labels remain
missing. See `scoring_definition.md`.
