# Theory Grounding

The central model follows Szeliski's learned motion-model idea: stack dense
motion fields from a sequence, subtract a mean field, and decompose the result
with SVD/PCA to obtain basis flow fields and temporal coefficients [Szeliski,
2010 draft, Sec. 8.2.2, printed pp. 403-404, PDF pp. 425-426].

The implemented classical baseline is:

```text
RGB video
-> deterministic frame decoding and resizing
-> optional body-centered alignment
-> dense optical flow
-> compact normalized flow fields
-> SVD/PCA motion basis
-> temporal coefficient features
-> Ridge score regression and irregularity heads
-> visual explanations
```

Optical flow is treated as a dense displacement-estimation problem under
brightness constancy and local smoothness assumptions [Szeliski, 2010 draft,
Sec. 8.4, printed pp. 409-414, PDF pp. 431-436]. The code uses OpenCV Farneback flow when
OpenCV is installed and a deterministic frame-difference fallback for smoke
tests. The fallback is not a replacement for the reported classical baseline; it
exists so tests can run in minimal Python environments.

Body-centered residual flow is motivated by translational alignment: global
image displacement can dominate the motion field, so this project compares
absolute flow against flow with estimated global translation removed [Szeliski,
2010 draft, Sec. 8.1, printed pp. 384-397, PDF pp. 406-419].

Temporal gait features use autocorrelation and Fourier summaries because gait is
approximately periodic in stable walking. This connects image motion to the
book's treatment of filtering and Fourier analysis [Szeliski, 2010 draft, Ch. 3,
printed pp. 99-203, PDF pp. 121-225; Sec. 3.4, printed pp. 138-169, PDF pp.
160-191].

Least-squares fitting appears in the Ridge-style regressor and in the SVD/PCA
projection step. Ridge is solved as an augmented least-squares system with
`numpy.linalg.lstsq`, avoiding the poorer conditioning of normal equations.
The motion basis uses a centered incremental SVD so clips do not need to be
stacked into one multi-gigabyte matrix [Szeliski, 2010 draft, App. A, printed
pp. 736-741, PDF pp. 757-763; App. B.2, printed pp. 759-766, PDF pp. 781-788].

All clips are resampled to the configured 20 FPS before flow extraction. FFT
features are reported in Hz using that sampling interval; this replaces the old
frame-rate-dependent bin-index calculation.

## Scientific Scope

The target is a simulation-derived gait quality score. It is not a clinical gait
score, not a human-biomechanics validation, and not proof that a policy is
dynamically optimal. RGB video may hide contact forces and internal controller
instability; simulator labels are not independent human judgements.
