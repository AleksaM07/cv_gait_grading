# Third-Party Notices

This repository is MIT licensed and does not vendor source from the inspected
reference repositories.

- Richard Szeliski, *Computer Vision: Algorithms and Applications*, 2010 draft:
  theoretical reference for dense motion, optical flow, and learned motion
  bases.
- `AleksaM07/mujoco-bipedal-joystick-walker`: CSV/video exports, transparent
  source-score definition, and conceptual host-replay behavior. The active
  recorder is an original minimal implementation; environment and policy code
  remain external because the inspected checkout has no license file.
- PECoP: conceptual AQA reference only; no license was found in the inspected
  checkout and no source was copied.
- CARE-PD and A. Vakanski rehabilitation-assessment repositories: conceptual
  references; their inspected code is MIT licensed and no source was copied.
- External gait datasets are cited in `references/CITATIONS.bib`; none are
  redistributed by this repository.
- The video-transfer baseline uses torchvision's R3D-18 implementation and
  official Kinetics-400 V1 pretrained weights. Torchvision is BSD-3-Clause
  licensed. The downloaded weight file and derived model stay in ignored local
  output directories and are not distributed with this repository.

Exact inspected commits and reuse decisions are recorded in
`datasets/README.md`, `reports/assets/docs/reference_repository_audit.md`, and
`reports/assets/docs/provenance.csv`.
