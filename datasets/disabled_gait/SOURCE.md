# DisabledGait provenance

- Source: https://data.mendeley.com/datasets/v6hy35ydch/2
- DOI: `10.17632/v6hy35ydch.2`
- Published dataset name: *Gait Dataset of Normal People and People with
  Disabilities* (DisabledGait v2)
- Reported license: CC BY 4.0
- Local purpose: real RGB walking videos grouped as `assistive`,
  `non_assistive`, and `normal`

The downloaded package contained 130 MP4 files plus 6,500 extracted JPG frames
and 6,500 YOLO TXT annotations. This project retains only 125 byte-unique
walking videos. Five duplicate copies are recorded in
`excluded_duplicates.csv`; frame images and detection annotations were not
retained because the gait pipeline consumes videos.

Videos were flattened into `videos/` and renamed as
`disabled_gait__<category>__<source-id>.mp4`. The H.264 stream was losslessly
copied into a clean MP4 without the damaged/unneeded AAC stream; video pixels
were not re-encoded. `manifest.csv` preserves the original relative path,
hashes, category, duration, frame rate, and dimensions.
