# Parallax
multimodal perception for spatial reasoning and active tracking


Parallax
│
├── Docker
│
├── ROS 2 workspace
│
├── Calibration package
│
├── Stereo package
│
├── Pan/Tilt package
│
├── LiDAR package
│
├── Fusion package
│
└── World Model package

Before you can meaningfully fuse stereo and LiDAR, you need to be comfortable with:

* camera intrinsics
* camera extrinsics
* homogeneous coordinates
* rigid transforms (SE(3))
* projection
* back-projection
* coordinate frame transforms

Those become utilities you’ll reuse throughout the project.
That package wouldn’t know anything about cameras yet. It would just answer questions like:

* Project a 3D point into image space.
* Back-project a pixel into a ray.
* Transform a point from camera → world.
* Transform world → camera.
* Compose transformations.
* Load calibration YAML.

---
relevant:
stereo_depth_demo/utils.py
 - camera access infrastructure 

stereo_depth_demo/arducam_camera.py
 - general opencv video capture wraper(mostly useful might need to use cvcuda/accelerated instead)
 - orientation logic not directly portable
 - need to verify AR0324 bayer pattern from actual driver-reported format than coping those register writes

stereo_depth_demo/4_calibration.py
 - actual stereo calibration, reads captured frames and passes to third party python stereovision package
 - implement this using OpenCV’s standard APIs which will gives you direct control over K, D, R, T, E, F, R1, R2, P1, P2, and Q.

