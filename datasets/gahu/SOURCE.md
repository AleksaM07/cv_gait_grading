# GaHu-Video provenance

- Source: https://data.mendeley.com/datasets/gprg4s73v4/1
- DOI: `10.17632/gprg4s73v4.1`
- Published dataset name: *GaHu-Video: Parametrization system for human gait
  recognition*
- Reported license: CC BY 4.0
- Local purpose: trimmed side-view walking clips

Ten source archives contained 440 H.264 AVI videos: 44 long `Originals` and
nine trimmed track/view variants for each of 44 subjects. The active local set
contains 395 walking-only clips:

- 44 long originals were excluded because they contain substantial empty,
  non-walking intervals between passes;
- one byte-identical track copy was removed;
- three `.dat` geometric-feature archives were not retained because this
  project learns from video.

The retained video stream was losslessly remuxed from AVI to MP4 with
`-c:v copy`; the damaged/unneeded AAC stream was omitted. Files were flattened
into `videos/` and named
`gahu__<subject>__track<1-3>_<left|center|right>.mp4`. `manifest.csv` preserves
the source path, subject, track/view identity, hashes, duration, frame rate,
and dimensions. Exclusions are recorded in `excluded_duplicates.csv`.
