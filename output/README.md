# Runtime outputs

This directory contains regenerable training and inference artifacts. Its
contents are ignored by Git except for this file and `.gitkeep` markers.

```text
models/       serialized trained models
predictions/  predictions, evaluation tables, JSON, and diagnostic figures
videos/       annotated/scored videos
logs/         command logs
```

Dataset payloads belong in their documented local roots, not here. See
`datasets/README.md` for origin and regeneration information.
