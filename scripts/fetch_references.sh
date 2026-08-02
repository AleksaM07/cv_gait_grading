#!/usr/bin/env bash
set -euo pipefail

mkdir -p _references

git clone --depth 1 https://github.com/AleksaM07/mujoco-bipedal-joystick-walker.git _references/mujoco-bipedal-joystick-walker
git clone --depth 1 https://github.com/Plrbear/PECoP.git _references/PECoP
git clone --depth 1 https://github.com/TaatiTeam/CARE-PD.git _references/CARE-PD
git clone --depth 1 https://github.com/avakanski/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises.git _references/rehab-assessment-deep
git clone --depth 1 https://github.com/avakanski/Rehabilitation-Assessment-through-Dimensionality-Reduction-and-Statistical-Modeling.git _references/rehab-assessment-statistical

for repo in _references/*; do
  echo "== ${repo}"
  git -C "${repo}" rev-parse HEAD
done
