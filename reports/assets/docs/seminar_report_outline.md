# Seminar Report Outline

1. Introduction
   Relevant theory: AQA framing and visual motion estimation. Figures: system
   overview. Tables: research questions and outputs.

2. Existing humanoid locomotion project
   Implementation files: `src/gait_aqa/data/import_walker_outputs.py`. Figures:
   walker CSV import flow. Tables: available telemetry fields.

3. Action Quality Assessment
   External citations: PECoP, CARE-PD, rehabilitation assessment papers. Figures:
   score-regression task diagram.

4. Relevant theory from Szeliski
   Book sections: Ch. 3, Ch. 4, Ch. 6, Ch. 8, Ch. 12, Ch. 14, App. A/B. Table:
   `reports/assets/docs/book_mapping.md`.

5. Optical flow and motion estimation
   Book sections: Sec. 8.1, 8.1.1, 8.4, 8.4.1. Implementation:
   `src/gait_aqa/vision/optical_flow.py`.

6. Learned motion models using SVD/PCA
   Book section: Sec. 8.2.2. Implementation:
   `src/gait_aqa/vision/motion_basis.py`. Figures: basis-flow fields.

7. Dataset generation in MuJoCo
   Implementation: `src/gait_aqa/reference_videos/render_mujoco.py` and
   `src/gait_aqa/data/build_manifest.py`. Tables: manifest schema.

8. Ground-truth score construction
   Implementation: `src/gait_aqa/labels/score_components.py`. Tables:
   component weights and threshold definitions.

9. Classical visual scoring system
   Implementation: `src/gait_aqa/models/classical_regressor.py`,
   `src/gait_aqa/training/train_classical.py`.

10. Modern video baseline
    Frozen R3D-18 Kinetics-400 backbone, unlabeled PCA/whitening, grouped Ridge
    head, and policy-level aggregation.

11. Experimental setup
    Implementation: `src/gait_aqa/data/split_dataset.py`,
    `src/gait_aqa/evaluation/metrics.py`.

12. Results
    Tables: fixed policy-held-out and nested policy CV MAE, RMSE, R2, Spearman,
    and pairwise ranking accuracy versus train-mean and classical baselines.

13. Failure analysis
    Figures: score-over-time plots, flow overlays, high-disagreement examples.

14. Limitations
    Include simulator-derived labels, camera/rendering bias, RGB-contact
    ambiguity, no clinical validity, and no demonstrated sim-to-real transfer.

15. Conclusion
    Summarize whether external RGB video can rank simulated humanoid gaits under
    the tested protocols.
