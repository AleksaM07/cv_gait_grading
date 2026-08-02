# Theory Grounding

The central model follows Szeliski's learned motion-model idea: stack dense
motion fields from a sequence, subtract a mean field, and decompose the result
with SVD/PCA to obtain basis flow fields and temporal coefficients [Szeliski,
2010 draft, Sec. 8.2.2, pp. TODO: verify page].

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
Sec. 8.4, pp. TODO: verify page]. The code uses OpenCV Farneback flow when
OpenCV is installed and a deterministic frame-difference fallback for smoke
tests. The fallback is not a replacement for the reported classical baseline; it
exists so tests can run in minimal Python environments.

Body-centered residual flow is motivated by translational alignment: global
image displacement can dominate the motion field, so this project compares
absolute flow against flow with estimated global translation removed [Szeliski,
2010 draft, Sec. 8.1, pp. TODO: verify page].

Temporal gait features use autocorrelation and Fourier summaries because gait is
approximately periodic in stable walking. This connects image motion to the
book's treatment of filtering and Fourier analysis [Szeliski, 2010 draft, Ch. 3,
pp. TODO: verify page].

Least-squares fitting appears in the Ridge-style regressor and in the SVD/PCA
projection step. The implementation follows the usual linear-algebra view of
least squares and low-rank subspaces [Szeliski, 2010 draft, App. A, pp. TODO:
verify page].

## Scientific Scope

The target is a simulation-derived gait quality score. It is not a clinical gait
score, not a human-biomechanics validation, and not proof that a policy is
dynamically optimal. RGB video may hide contact forces and internal controller
instability; simulator labels are not independent human judgements.
