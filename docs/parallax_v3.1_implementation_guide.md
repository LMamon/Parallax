# Parallax v3.1 --- Implementation Guide

## Objective

Implement Parallax v3.1 as an aggressive integration of the
already-proven pieces:

-   real AR0234 stereo hardware and calibration from Parallax;
-   the ROS 2 / Isaac ROS spatial spine already proven in **RXSIM** https://github.com/LMamon/RXSIM/tree/main/rosgz/src;
-   NanoOWL, EfficientViT-SAM, Object3D, tracking, and sensor-fusion
    behavior from **Parallax** https://github.com/LMamon/Parallax/tree/spatial-mapping;
-   nvblox for persistent TSDF/3D ESDF;
-   OMPL for 3D path planning;
-   `sllidar_ros2` for the C1;
-   Foxglove for interaction and visualization.

This guide deliberately targets **5--10 coherent commits**. The
preferred implementation is **8 commits**.

The rule throughout is:

> **Integrate first. Configure before adapting. Adapt before writing. Do
> not rebuild infrastructure that ROS 2, Isaac ROS, or an established
> ROS package already owns.**

Each commit should leave the branch buildable and should have a concrete
validation gate.

------------------------------------------------------------------------

# 0. Before Coding

## Inputs already available

Use these as source material rather than rediscovering the architecture:

``` text
docs/parallax_v3.1_implementation_guide.md
docs/parallax_v3.1_references.md
https://github.com/LMamon/RXSIM/tree/main/rosgz/src
current Parallax https://github.com/LMamon/Parallax/tree/spatial-mapping implementation
existing stereo calibration outputs in config/
existing NanoOWL / SAM / Object3D / tracking code
```

## Working assumptions

Target environment:

``` text
Jetson Orin Nano 8 GB
JetPack 6.2.1 / L4T R36.4.7
One docker container/file
ROS 2 Humble
Isaac ROS release 3.2
AR0234 SBS: 3840x1200
per-eye resolution: 1920x1200
RPLIDAR C1
Foxglove on host
```

Do not silently switch to newer Isaac ROS documentation while
implementing. Validate component names, parameters, messages, and
services against the installed release-3.2 packages.
the idea is to run start with just ./scripts/run.sh and test with ./scripts/test.sh so the user can one line run this or one line make sure it will build/run similar to how its set up in the main branch.

## Branch

Suggested:

``` bash
git switch -c v3.1
```

## Target workspace

``` text
ros2_ws/
└── src/
    ├── bringup/
    │   ├── launch/
    │   └── config/
    │
    ├── camera/
    │   ├── launch/
    │   ├── config/
    │   └── src/
    │
    ├── perception/
    │   ├── launch/
    │   ├── config/
    │   └── src/
    │
    ├── fusion/
    │   └── src/
    │
    ├── planning/
    │   ├── config/
    │   ├── include/
    │   └── src/
    │
    └── interfaces/
        └── only if a genuinely custom interface is required
```

Do not create packages merely to mirror the old Parallax directory
structure.

------------------------------------------------------------------------

# Commit 1 --- Establish the ROS/Isaac Runtime

Suggested commit:

``` text
build: establish ROS 2 and Isaac ROS v3 runtime
```

## Goal

Get a reproducible container and ROS workspace running before porting
application code.

## Work

Adapt the existing Parallax Docker setup rather than copying RXSIM's
Docker environment wholesale.

The runtime needs:

-   ROS 2 Humble;
-   Isaac ROS 3.2 packages needed by the image/stereo/cuVSLAM/nvblox
    path;
-   OMPL;
-   Foxglove bridge;
-   `sllidar_ros2`;
-   CUDA/TensorRT/PyTorch dependencies needed by NanoOWL/SAM;
-   colcon/build tooling;
-   Parallax workspace.

Compose should provide:

-   NVIDIA runtime;
-   host network;
-   host IPC where required;
-   `/dev/video0`;
-   C1 serial device;
-   configuration/log/map volumes;
-   ROS domain configuration.

Prefer explicit devices over:

``` yaml
privileged: true
```

unless an actual hardware requirement proves it necessary.

Create the initial `bringup` package and one top-level launch entry
point.

## Do not do yet

Do not port:

-   Object3D;
-   mapping code;
-   neural perception;
-   planner;
-   custom runtime/DAG;
-   old RPLIDAR SDK integration.

## Gate

Inside the container:

``` bash
ros2 pkg list
ros2 node list
```

must work.

The workspace must:

``` bash
colcon build
source install/setup.bash
```

cleanly.

`docker compose up` should start the base ROS graph without
hardware-processing failures caused by missing runtime dependencies.

------------------------------------------------------------------------

# Commit 2 --- Integrate the Physical Stereo Pipeline

Suggested commit:

``` text
feat: integrate AR0234 stereo ingress and preprocessing
```

## Goal

Turn the physical SBS camera into standard calibrated rectified stereo
ROS products.

## Target graph

``` text
/dev/video0
    |
    v
v4l2_camera
    |
    v
/stereo/stitched_image
    |
    +----------> left CropNode
    |                |
    |                v
    |          left CameraInfo
    |                |
    |                v
    |           RectifyNode
    |
    +----------> right CropNode
                     |
                     v
               right CameraInfo
                     |
                     v
                RectifyNode
```

## Camera

Start from the known working physical format:

``` text
3840 x 1200 SBS
38 FPS target
```

Do not change physical capture mode merely to simplify ROS integration.

## SBS split

Left:

``` text
x = 0
y = 0
width = 1920
height = 1200
```

Right:

``` text
x = 1920
y = 0
width = 1920
height = 1200
```

Both outputs must preserve the source-frame timestamp.

## Calibration

Convert/load the existing calibration into standard per-eye
`CameraInfo`.

Left:

``` text
K_left
D_left
R1
P1
```

Right:

``` text
K_right
D_right
R2
P2
```

Do not invent calibration from the stitched frame.

Do not put `Q` into `CameraInfo`.

Use ROS optical-frame conventions.

Suggested frames:

``` text
base_link
left_camera_optical_frame
right_camera_optical_frame
```

Add static physical stereo transforms where required by the downstream
graph.

## Rectification

Use the Isaac ROS image pipeline where supported by the installed
release.

Do not port the custom VPI rectification implementation unless the Isaac
path demonstrably cannot satisfy the camera.

## NITROS boundary

Do not spend this commit chasing theoretical zero-copy capture.

A deliberate ingress boundary:

``` text
V4L2 / standard ROS image
        ->
Isaac/NITROS accelerated graph
```

is acceptable.

## Inspect before moving on

``` bash
ros2 topic list
ros2 topic hz <left-image-topic>
ros2 topic hz <right-image-topic>
ros2 topic echo <left-camera-info-topic> --once
ros2 topic echo <right-camera-info-topic> --once
```

Verify matching timestamps for each stereo pair.

## Foxglove gate

Display both rectified images.

Required:

-   correct left/right split;
-   no accidental eye swap;
-   rectification looks correct;
-   no obvious stretching/cropping error;
-   stable expected rate;
-   valid CameraInfo;
-   correct frame IDs.

Do not proceed until this is stable.

------------------------------------------------------------------------

# Commit 3 --- Restore the RXSIM Spatial Spine on Real Hardware

Suggested commit:

``` text
feat: integrate stereo depth cuVSLAM and nvblox
```

## Goal

This is the commit where v3.1 becomes a functioning spatial system.

Integrate:

``` text
rectified stereo
    |
    +----> stereo disparity/depth
    |
    +----> cuVSLAM
              |
              v
             TF

depth + TF
    |
    v
nvblox
    |
    +----> TSDF
    +----> mesh
    `----> 3D ESDF
```

## Stereo

Use the release-3.2 Isaac stereo components.

Initial disparity configuration can use the already-selected maximum
disparity of 128.

If OFA is unavailable on the Nano, use the supported CUDA path rather
than building another stereo matcher.

Point-cloud publication is optional for the mapping path.

Preferred:

``` text
disparity -> depth -> nvblox
```

Generate a point cloud only when useful for debugging/visualization.

## cuVSLAM

Start from the actual RXSIM configuration/remappings.

Known RXSIM relationship:

``` text
visual_slam/image_0       -> left rectified image
visual_slam/camera_info_0 -> left CameraInfo

visual_slam/image_1       -> right rectified image
visual_slam/camera_info_1 -> right CameraInfo
```

Use:

``` text
num_cameras: 2
rectified_images: true
```

Preserve synchronized stereo timestamps.

Do not carry simulation IMU/PX4 assumptions into the real graph unless
the real system supplies them.

## TF

Confirm the actual resulting tree with:

``` bash
ros2 run tf2_tools view_frames
```

The graph must provide the transforms nvblox needs from its global/map
frame to the camera/depth frame.

Do not add duplicate static transforms for transforms already owned by
cuVSLAM.

## nvblox

Start from the RXSIM config, especially:

``` yaml
mapping_type: "static_tsdf"
global_frame: "map"
odom_frame: "odom"
base_frame: "base_link"
use_depth: true
use_color: false
use_lidar: false
esdf_mode: "3d"
```

Keep the C1 out of this path.

Use the existing RXSIM update rates as initial values; tune later.

## Foxglove

Show:

-   TF;
-   rectified image;
-   depth if useful;
-   cuVSLAM path/pose;
-   nvblox mesh.

## Gate

Move the physical camera rig around.

Required:

1.  cuVSLAM pose changes correctly.
2.  TF remains resolvable.
3.  stereo depth is metric and plausible.
4.  nvblox accumulates a persistent scene.
5.  the scene stays in the world while the rig moves.
6.  3D ESDF is enabled.
7.  a known-observed ESDF query produces meaningful values.

At this commit, tag mentally:

``` text
PARALLAX V3 SPATIAL SPINE WORKING
```

Do not begin porting the old custom mapping path after this succeeds.

------------------------------------------------------------------------

# Commit 4 --- Integrate C1 and Manual 3D Planning

Suggested commit:

``` text
feat: add lidar ingress and ESDF-backed 3D planning
```

This is the largest commit. Split it into two commits if necessary.

If split:

``` text
feat: integrate RPLIDAR C1 ROS driver
feat: add ESDF-backed OMPL planning
```

That produces a 9-commit plan instead of 8.

## Part A --- C1

Use:

``` text
RPLIDAR C1
    ->
sllidar_ros2
    ->
sensor_msgs/LaserScan
```

Add its physical TF/extrinsics.

Do not port raw serial handling unless `sllidar_ros2` fails an actual C1
requirement.

Gate:

-   scan visible;
-   correct orientation;
-   correct range scale;
-   correct frame;
-   no coupling into nvblox.

## Part B --- Planner

Create the thin integration seam Parallax actually needs.

Suggested planning components:

``` text
planning/
  include/planning/
    esdf_cache.hpp
    planner.hpp
    planning_config.hpp

  src/
    esdf_cache.cpp
    planner.cpp
    planner_node.cpp
```

Do not build a generic planning framework.

## Planner input

Use a standard manual goal interface where practical.

Goal must resolve to:

``` text
geometry_msgs/PoseStamped
frame = map
```

Foxglove can publish the goal.

## Start

Obtain current rig pose through TF.

Failure to obtain TF:

``` text
planning request fails
```

Never:

``` text
start = (0, 0, 0)
```

## Planning AABB

Construct:

``` text
min(start, goal) - margin
max(start, goal) + margin
```

Request the nvblox 3D ESDF for that region using the release-3.2 service
schema.

Use asynchronous ROS service handling.

Do not block the node executor with an unnecessary nested spin.

## EsdfCache

The cache should contain only planner-facing data:

``` text
origin
voxel size
dimensions
layout/strides
distance values
observed semantics
```

Validate the exact `Float32MultiArray` indexing against the installed
release.

Treat nvblox's unobserved sentinel correctly.

Unknown/unobserved:

``` text
invalid
```

## OMPL

Initial space:

``` cpp
ompl::base::RealVectorStateSpace(3)
```

Initial planner:

``` cpp
ompl::geometric::RRTConnect
```

Validity:

``` text
observed
AND
distance > robot_radius + safety_margin
```

Set state-space bounds from the current planning AABB.

Configure OMPL's motion-validity resolution relative to:

-   ESDF voxel size;
-   robot radius;
-   minimum obstacle thickness worth respecting.

Do not rely only on endpoint validity.

## Retry

If planning fails:

``` text
expand AABB
    ->
request larger ESDF snapshot
    ->
retry
```

Keep retry policy simple and bounded.

## Output

Publish:

``` text
nav_msgs/Path
frame_id = map
```

This is a geometric path, not a time-parameterized trajectory.

## Gate

From Foxglove:

1.  publish a manual XYZ goal;
2.  planner obtains current TF pose;
3.  ESDF snapshot succeeds;
4.  start and goal validate;
5.  OMPL runs;
6.  path appears in Foxglove.

Test:

-   clear straight-line goal;
-   goal near an obstacle;
-   blocked straight line requiring a detour;
-   goal in unknown space;
-   unavailable TF;
-   goal outside first AABB detour requiring expansion.

------------------------------------------------------------------------

# Commit 5 --- Port NanoOWL and EfficientViT-SAM as Managed Capabilities

Suggested commit:

``` text
feat: integrate managed neural perception
```

## Goal

Bring neural perception into ROS without porting the old execution
runtime around it.

## Graph

``` text
rectified / ISP image
    |
    v
NanoOWL
    |
    v
2D detection
    |
    v
EfficientViT-SAM
    |
    v
segmentation
```

Keep model-specific implementation where it is useful.

Replace custom transport/orchestration with ROS mechanisms.

## Resource behavior

The neural stack must support actual resource ownership.

Effective requirement:

``` text
neural_required =
    explicit_neural_request
    OR
    tracking_requires_neural
```

For this commit, implement at least:

``` text
activate
    ->
models/resources available
    ->
inference runs

zero consumers
    ->
inference stops
    ->
GPU-heavy resources released
```

If a clean ROS lifecycle-node implementation fits the Python/model
stack, use it.

If not, use the smallest ROS-native service/component boundary
necessary. Do not build a generic custom lifecycle framework.

## Verify memory

Record GPU/system memory before activation, during inference, and after
final deactivation.

The point is not merely:

``` text
no detections are published
```

The point is:

``` text
unused neural resources are no longer occupying scarce memory
```

## Gate

1.  Start neural perception.
2.  Receive detections.
3.  Run segmentation.
4.  Stop the final consumer.
5.  Confirm inference stops.
6.  Confirm model resources can be released.
7.  Restart.
8.  Confirm perception recovers cleanly.

------------------------------------------------------------------------

# Commit 6 --- Port Object3D, DCF Tracking, and LiDAR Evidence

Suggested commit:

``` text
feat: integrate metric Object3D tracking and sensor evidence
```

## Goal

Bring over the application behavior that makes Parallax more than a
collection of ROS nodes.

## Graph

``` text
NanoOWL detection
       +
SAM mask
       +
stereo depth
       |
       v
metric Object3D
       |
       v
DCF / extended tracking
       |
       +----> ROS-facing Detection3D
       |
       +----> Foxglove 3D geometry
       |
       +----> semantic target resolver
       |
C1 ----+----> optional metric evidence / association
```

## Object3D

Preserve useful internal state:

-   semantic identity;
-   provenance;
-   stereo metric evidence;
-   segmentation-supported extent;
-   tracking state;
-   LiDAR evidence;
-   geometry method/state;
-   confidence/association state.

Do not force all internal state into a standard ROS message.

Use `vision_msgs/Detection3DArray` or another appropriate standard
representation as the ROS-facing interoperability product.

## Geometry

The same authoritative metric geometry must drive:

``` text
3D Foxglove box
image reprojection
tracking
target selection
approach-point generation
```

Do not independently calculate one box for visualization and another for
planning.

## DCF dependency behavior

Tracking implicitly requires neural perception.

Required sequence:

``` text
explicit neural ON
tracking ON
explicit neural OFF
```

Result:

``` text
neural remains ON
```

Then:

``` text
tracking OFF
```

If no other consumer exists:

``` text
neural resources release
```

The tracker must not directly tear down neural resources owned by
another active capability.

## C1 evidence

Port only the useful association/evidence logic.

Do not port raw serial transport.

## Gate

-   detect object;
-   segment object;
-   obtain metric depth;
-   produce world-resolved Object3D;
-   display its 3D extent;
-   track it;
-   validate DCF/neural dependency behavior;
-   associate C1 evidence where geometrically applicable.

------------------------------------------------------------------------

# Commit 7 --- Add Semantic Target-to-Path Integration

Suggested commit:

``` text
feat: connect semantic targets to 3D planning
```

## Goal

Connect perception to the already-working planner without creating a
second planning path.

## Flow

``` text
semantic request
    |
    v
target lookup
    |
    v
tracked Object3D
    |
    v
approach-point generation
    |
    v
PoseStamped goal in map
    |
    v
existing manual-goal planner
    |
    v
nav_msgs/Path
```

Manual and semantic planning must converge before ESDF/OMPL.

Do not duplicate planner logic.

## Semantic request

Keep the application interface small.

If a standard ROS action/service expresses the request cleanly, use it.

If not, add the minimum custom interface required for semantics such as:

``` text
path target=cup
```

Do not turn this into a general natural-language command framework.

## Target resolution

Resolve the semantic label to an active/tracked Object3D.

Handle explicitly:

-   no matching target;
-   multiple matching targets;
-   stale target;
-   target without valid metric geometry.

For the first demonstration, a deterministic selection rule is enough.

## Approach point

Do not use the object centroid as the goal.

Generate candidate free points using:

-   object extent/AABB;
-   rig radius;
-   safety margin;
-   desired standoff;
-   ESDF clearance;
-   optional LOS.

Select a valid candidate and send it through the existing planning
input.

## Gate

From Foxglove:

``` text
path to cup
```

must produce:

``` text
semantic resolution
-> Object3D
-> free approach goal
-> ESDF request
-> OMPL
-> nav_msgs/Path
-> Foxglove visualization
```

No manually entered XYZ is required for this test.

------------------------------------------------------------------------

# Commit 8 --- Harden Bringup, Lifecycle, Foxglove, and Demo

Suggested commit:

``` text
chore: harden v3.1 bringup and demonstration
```

## Goal

Turn the integrated branch into a reproducible demonstration rather than
a collection of individually working launch commands.

## Bringup

One top-level launch should establish the normal system graph.

Avoid arbitrary `TimerAction` sequencing copied from simulation.

Prefer:

-   lifecycle readiness;
-   component state;
-   actual ROS dependency behavior;
-   explicit failure reporting.

## Runtime audit

For every expensive subsystem, record:

``` text
Who explicitly requests it?
What capabilities implicitly require it?
What keeps it resident?
What happens when the final consumer disappears?
What state remains after stop?
How is stale output cleared?
Can it restart cleanly?
```

Audit:

-   camera;
-   stereo preprocessing;
-   depth;
-   cuVSLAM;
-   nvblox;
-   NanoOWL;
-   SAM;
-   DCF tracker;
-   C1;
-   Object3D;
-   planner;
-   visualization products;
-   recording.

Do not force every node off merely because it can be stopped. The
objective is coherent resource ownership.

## Foxglove layout

Save a layout containing:

### 3D

-   TF;
-   nvblox mesh;
-   cuVSLAM trajectory;
-   Object3D boxes;
-   C1 scan;
-   goal;
-   planned path;
-   optional planning AABB.

### Image

-   rectified/ISP image;
-   detection overlay;
-   SAM mask;
-   projected Object3D geometry.

### Controls

-   manual goal publication;
-   semantic path request;
-   useful capability/lifecycle controls.

## Recording

Add rosbag2 configuration for the important reproducibility/debug
products.

Do not blindly record every high-bandwidth topic.

## Cleanup

Remove or retire obsolete infrastructure that v3.1 no longer uses:

-   custom mapping runtime pieces replaced by nvblox;
-   raw C1 driver integration;
-   redundant visualization transport;
-   redundant execution/synchronization machinery;
-   abandoned launch/config experiments.

Do not delete useful old implementation until the replacement has passed
its gate.

## Final gate

A clean checkout on the target Jetson should be able to reach:

``` bash
docker compose build
docker compose up
```

followed by the expected Parallax ROS graph.

------------------------------------------------------------------------

# Commit Budget

Preferred:

``` text
1  build: establish ROS 2 and Isaac ROS v3 runtime
2  feat: integrate AR0234 stereo ingress and preprocessing
3  feat: integrate stereo depth cuVSLAM and nvblox
4  feat: add lidar ingress and ESDF-backed 3D planning
5  feat: integrate managed neural perception
6  feat: integrate metric Object3D tracking and sensor evidence
7  feat: connect semantic targets to 3D planning
8  chore: harden v3.1 bringup and demonstration
```

If Commit 4 becomes too large:

``` text
4  feat: integrate RPLIDAR C1 ROS driver
5  feat: add ESDF-backed OMPL planning
```

and renumber the remaining commits.

That gives **9 commits**.

Do not split work merely to create small commits. Split when a commit
has two independently testable architectural gates.

------------------------------------------------------------------------

# Definition of Done

The branch is done when the following works as one system.

## Bringup

``` text
docker compose up
```

starts the expected ROS graph.

## Geometry

``` text
AR0234
 -> SBS split
 -> calibrated rectified stereo
 -> depth
 -> cuVSLAM pose/TF
 -> nvblox persistent TSDF
 -> 3D ESDF
```

## Manual planning

``` text
Foxglove XYZ goal
 -> current map pose
 -> ESDF
 -> OMPL
 -> collision-aware nav_msgs/Path
```

## Semantic perception

``` text
NanoOWL
 -> SAM
 -> metric Object3D
 -> DCF tracking
 -> optional C1 metric evidence
```

## Semantic planning

``` text
"path to cup"
 -> tracked cup
 -> free approach point
 -> same ESDF/OMPL planner
 -> path
```

## Resource behavior

``` text
explicit neural OFF + tracking ON
 -> neural stays active

tracking OFF + no neural consumers
 -> neural resources release
```

Equivalent dependency ownership should hold across the rest of the
system.

## Visualization

Foxglove can inspect:

-   stereo imagery;
-   TF;
-   localization;
-   nvblox reconstruction;
-   Object3D;
-   C1;
-   goals;
-   planned paths.

------------------------------------------------------------------------

# Debugging Order

When integration fails, debug from upstream to downstream.

## Camera failure

``` text
device
-> V4L2
-> stitched ROS image
-> crops
-> CameraInfo
-> rectification
```

Do not debug cuVSLAM while the stereo contract is wrong.

## cuVSLAM failure

Check:

``` text
rectified images
timestamps
CameraInfo
frame IDs
stereo extrinsics
input format
sync threshold
```

before changing SLAM parameters.

## nvblox failure

Check:

``` text
depth validity
depth CameraInfo
TF connectivity
global frame
depth frame
timestamps
```

before changing mapper internals.

## Planner failure

Check:

``` text
TF start
goal frame
ESDF service success
ESDF observed values
cache indexing
start validity
goal validity
AABB bounds
motion validation
```

before changing OMPL algorithms.

## Semantic path failure

Check:

``` text
detection
-> segmentation
-> depth support
-> Object3D
-> world transform
-> tracking
-> approach point
-> planner
```

Do not debug semantic target resolution by modifying the planner if
manual planning already works.

------------------------------------------------------------------------

# Scope Guard

If implementation starts expanding beyond 10 commits because new
infrastructure is being designed, stop and ask:

> Is this actually a missing Parallax capability, or are we rebuilding
> something the ROS ecosystem already owns?

The expected v3.1 effort is integration-heavy and invention-light.

The success metric is not how much code survives from `main`.

The success metric is how quickly the known pieces become one coherent,
reproducible, resource-aware system.
