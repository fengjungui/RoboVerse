# Real2Sim

The general idea of Real2Sim is to reconstruct the static scene and generate scene-aligned digital asset for robotic manipulation.

## Prerequisite
1. supersplat
2. colmap / glomap
3. gsplat
4. stablenormal
5. 2dgs

## Process
1. Take video
2. Run colmap
3. Run gsplat
4. Mesh extraction (StableNormal, 2dgs)
5. Recenter and Reorientation
6. Segmentation for 3DGS and mesh
7. Assign ID for 3DGS
8. Fix Kinemics & Dynamics parameters
9. Construct URDF
10. Alignment: coordinate & scale (between mesh, 3DGS and physics engines)
11. Physics-awared rendering (fk, ik, mujoco) (for vla)
12. load urdf for policy (for rl and imitation learning)
