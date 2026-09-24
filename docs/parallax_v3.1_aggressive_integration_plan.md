# Parallax v3.1 --- Aggressive Integration Plan

## Purpose

Parallax v3.1 is not a port of the current custom runtime into ROS 2,
and it is not another RXSIM. It is the integration of the
real-hardware/perception work proven in Parallax with the ROS 2 / Isaac
ROS spatial stack already proven in RXSIM.

The project itself is the integrated system. No individual detector,
mapper, tracker, planner, transport layer, or middleware component needs
to be uniquely Parallax.

> **Reuse existing robotics infrastructure aggressively. Configure
> before adapting. Adapt before writing. Keep Parallax focused on
> integrating capabilities into one coherent system.**

The migration stays runnable throughout. Get a minimal real-hardware
ROS/Isaac spine alive early, then add capabilities to that running
system.

## 1. Architectural correction

The earlier plan still assumed Parallax needed to define too much
infrastructure. ROS 2 and its ecosystem already provide the mechanisms
for topic transport, standard messages, timestamps, QoS, frame IDs, TF,
calibration transport, coordinate conventions, units, services/actions,
parameters, lifecycle, composition, launch, recording, introspection,
diagnostics, and visualization interoperability.

Isaac ROS already supplies accelerated implementations for major parts
of the geometry pipeline.

Therefore v3.1 is primarily an **integration problem**, not a
middleware-design problem.

For every requirement:

1.  Does ROS 2 already provide the mechanism?
2.  Does Isaac ROS or another established ROS package already provide
    the implementation?
3.  Can configuration/remapping solve it?
4.  Does Parallax only need a thin adapter?
5.  Only then: does Parallax need new code?

Classify work as **REUSE**, **CONFIGURE**, **ADAPT**, or **WRITE**.
`WRITE` should be the minority.

## 2. RXSIM → main → v3.1

**RXSIM:** proved that the ROS 2 / Isaac ROS spatial components could
interoperate: stereo, cuVSLAM, nvblox, TF, topics, Docker, and Foxglove.

**Parallax main:** proved the real-hardware/perception side and exposed
the abstractions that matter by owning much more of capture, products,
provenance, synchronization, stereo/depth, perception, fusion, Object3D,
tracking, mapping experiments, visualization, and execution behavior.

**Parallax v3.1:** combines those lessons without preserving
infrastructure merely because Parallax once implemented it.

``` text
real sensors
    + ROS 2 infrastructure
    + Isaac ROS accelerated geometry
    + NanoOWL / EfficientViT-SAM
    + Parallax tracking / Object3D / fusion
    + nvblox world representation
    + OMPL planning
    + Foxglove interaction
    = PARALLAX v3.1
```

The integration is the product.

## 3. Integration ledger

  Capability               Existing basis                      v3.1 action
  ------------------------ ----------------------------------- --------------------
  AR0234 capture           working Parallax V4L2 path          ADAPT
  SBS split                ROS/Isaac image processing          CONFIGURE / REUSE
  Calibration              existing calibration + CameraInfo   ADAPT
  Rectification            Isaac ROS image pipeline            REUSE
  Stereo depth             Isaac ROS stereo                    REUSE
  Localization             Isaac ROS cuVSLAM                   REUSE RXSIM
  TF                       ROS 2 tf2                           REUSE
  Persistent map           nvblox                              REUSE RXSIM
  TSDF / 3D ESDF           nvblox                              CONFIGURE / REUSE
  Detection                NanoOWL                             ADAPT
  Segmentation             EfficientViT-SAM                    ADAPT
  DCF tracking             Parallax                            ADAPT
  Object3D                 Parallax                            ADAPT
  ROS 3D detections        vision_msgs                         ADAPT
  C1 driver                sllidar_ros2                        REUSE
  LiDAR evidence           Parallax                            ADAPT
  3D planner               OMPL                                REUSE
  ESDF → OMPL validity     missing seam                        WRITE thin adapter
  Manual goal              standard ROS geometry interface     REUSE
  Path output              nav_msgs/Path                       REUSE
  Semantic target → goal   Parallax behavior                   WRITE / ADAPT
  Runtime lifecycle        ROS 2 mechanisms where supported    REUSE / ADAPT
  Recording                rosbag2                             REUSE
  Visualization            Foxglove                            REUSE

New custom infrastructure requires justification.

## 4. Target graph

``` text
AR0234 SBS
    |
    v
V4L2 -> split -> calibration -> rectify
    |                         |
    |                         +------> cuVSLAM ------> TF / pose
    |
    +------> stereo disparity/depth ------> nvblox
    |                                         |
    |                                     TSDF / ESDF
    |
    +------> NanoOWL -> EfficientViT-SAM
                         |
                         v
                  Object3D / tracking
                         |
                         v
                 semantic target
                         |
                         v
                  approach point
                         |
                         +-------------------+
                                             |
current pose + nvblox ESDF ------------------+
                                             v
                                         OMPL R^3
                                             |
                                             v
                                      nav_msgs/Path
                                             |
                                             v
                                         Foxglove

RPLIDAR C1 -> sllidar_ros2 -> LaserScan
                           -> Parallax metric evidence / association
                           -> Foxglove
```

## 5. ROS contracts are the contracts

Do not create parallel Parallax definitions for concepts ROS already
defines.

Use standard ROS interfaces wherever practical: `sensor_msgs/Image`,
`sensor_msgs/CameraInfo`, `sensor_msgs/LaserScan`, TF2, standard
geometry/odometry messages, `nav_msgs/Path`, `vision_msgs` where
appropriate, ROS headers/clocks, parameters, services/actions, QoS,
lifecycle, and launch.

The job is to make producers and consumers conform to existing ROS/Isaac
contracts. Prefer remapping/configuration over custom protocols.

## 6. RXSIM is the spatial reference

Reuse the known-good RXSIM relationships rather than rediscovering them.

Known cuVSLAM remappings included:

``` text
/visual_slam/image_0       -> /stereo/left
/visual_slam/camera_info_0 -> /stereo/left/camera_info
/visual_slam/image_1       -> /stereo/right
/visual_slam/camera_info_1 -> /stereo/right/camera_info
```

RXSIM nvblox already used a static TSDF, `map`/`odom`/`base_link`, depth
input, and explicit `esdf_mode: "3d"`.

Remove simulation-specific
Gazebo/PX4/MicroXRCEAgent/sim-clock/TimerAction machinery. Replace
simulated ingress with the physical AR0234 while disturbing downstream
contracts as little as possible.

## 7. Physical stereo boundary

The camera produces one synchronized 3840×1200 SBS frame. Split it into
1920×1200 left/right images before stereo processing.

Rules:

-   both eyes retain the same physical source timestamp;
-   matching CameraInfo corresponds to each eye;
-   existing K/D/R/P calibration remains authoritative;
-   use R1/P1 for left and R2/P2 for right;
-   Q is not invented as a CameraInfo field;
-   use ROS optical-frame conventions;
-   do not derive two eye calibrations from a synthetic stitched
    calibration;
-   one deliberate V4L2 CPU→GPU/NITROS ingress boundary is acceptable;
-   do not force Argus if the known-good V4L2 path is more reliable.

## 8. TF and coordinates

ROS TF is authoritative. Do not maintain a second world-coordinate
system inside Parallax.

Conceptually:

``` text
map -> odom -> base_link
                  |-- left_camera_optical_frame
                  |-- right_camera_optical_frame
                  `-- lidar_frame
```

Exact ownership follows the installed cuVSLAM/Isaac configuration and
physical extrinsics.

All planning inputs resolve to `map`: current pose, goals, Object3D
geometry, ESDF requests, and output paths. Missing required TF is an
explicit failure; never substitute `(0,0,0)`.

## 9. Capability-driven runtime

User intent requests capabilities. Active capabilities imply
dependencies. Dependencies determine which resources remain active.

``` text
USER INTENT
  map / detect / track / path XYZ / path object
        |
        v
CAPABILITY DEPENDENCIES
        |
        v
ROS / ISAAC COMPONENT STATE
        |
        v
CPU / GPU / SENSOR RESOURCES
```

Before writing a custom manager, use ROS lifecycle, components,
services/actions, parameters, and launch mechanisms wherever they fit.
Do not recreate the old Parallax runtime under another name.

A subsystem remains active while at least one active capability requires
it.

Examples:

-   stopping planning does not stop nvblox if mapping remains active;
-   stopping mapping does not stop depth if Object3D still needs it;
-   stopping explicit detection does not unload neural perception while
    DCF tracking needs it;
-   stopping the final neural consumer allows neural resources to be
    released;
-   hiding an expensive visualization product is not automatically
    equivalent to stopping its source product.

## 10. Neural lifecycle

NanoOWL and EfficientViT-SAM are scarce resources on the 8 GB Nano.

Neural perception remains required when:

``` text
explicit_neural_request OR dcf_tracking_active
```

  Explicit neural   DCF tracking   Neural             Tracker
  ----------------- -------------- ------------------ ---------
  OFF               OFF            OFF / releasable   OFF
  ON                OFF            ON                 OFF
  OFF               ON             ON implicitly      ON
  ON                ON             ON                 ON

Initial deterministic policy: **zero consumers → stop inference →
release neural resources**.

Possible states are UNLOADED, INACTIVE/WARM, and ACTIVE, but warm
retention can wait until reload latency proves it useful.

Also keep these operations distinct:

``` text
stop producing detections
!= clear tracker state
!= clear visualization
!= unload neural models
```

Apply the same distinction system-wide: stopping mapping is not
resetting a map; stopping planning is not deleting the last path; hiding
a visualization is not stopping its producer.

## 11. Mapping

Retire custom map infrastructure when nvblox already owns the
requirement.

nvblox owns persistent TSDF state, integration, mesh generation, ESDF
generation, map updates, and ROS-facing spatial products. Parallax
consumes them.

The `spatial-mapping` work remains useful because it exposed the
geometry/planning requirements; it does not need to remain the
implementation.

## 12. 3D planning

Planning is:

``` text
current localized pose + goal in map + 3D ESDF
    -> collision-free 3D path
```

Use OMPL `RealVectorStateSpace(3)` and `RRTConnect` initially. Do not
introduce SE(3) until orientation affects validity. Do not use Nav2 as
the arbitrary 3D planner, and do not introduce MoveIt2 merely to wrap
OMPL for a free point/sphere.

Initial validity:

``` text
observed && esdf_distance > robot_radius + safety_margin
```

Unknown is invalid. Motion-validation resolution must prevent edges from
jumping through thin obstacles.

Goal-triggered flow:

``` text
goal
 -> transform to map
 -> current pose from TF
 -> start/goal AABB + margin
 -> nvblox 3D ESDF request
 -> local CPU EsdfCache
 -> validate start/goal
 -> OMPL RRTConnect
 -> nav_msgs/Path
```

AABB is an optimization/detail, not the architecture. Expand it and
retry when a tight search region may have excluded a detour.

## 13. Object3D and semantic goals

Keep `Object3D` as the richer internal semantic/fusion state. Use
standard ROS representations such as `Detection3DArray` for
interoperability where appropriate.

One authoritative metric Object3D geometry should drive 3D
visualization, projected image overlays, target selection, tracking, and
approach-point generation.

Semantic path flow:

``` text
"path to cup"
 -> resolve cup to tracked Object3D
 -> choose reachable free approach point
 -> NavigationGoal
 -> same planner used by manual goals
```

Do not default to the object centroid because the object is occupied.
Approach points may use object extent/AABB, rig radius, safety margin,
preferred standoff, ESDF clearance, and LOS.

## 14. RPLIDAR C1

Use:

``` text
RPLIDAR C1 -> sllidar_ros2 -> sensor_msgs/LaserScan
```

Keep it independent from volumetric nvblox mapping. Parallax-specific
value begins after standard transport: metric evidence, association,
proximity, validation, and possible future watchdog behavior.

## 15. Foxglove

Foxglove is the operator/debug interface and consumes normal ROS
products.

3D panel: TF, nvblox map/mesh, cuVSLAM trajectory, Object3D boxes, C1
scan, goals, path, optional planning AABB.

Image panel: rectified/ISP image, NanoOWL detections, SAM masks,
projected Object3D geometry.

Interaction: manual XYZ goal, semantic path request, state/topic
inspection, and useful runtime/lifecycle controls.

Save a final layout for the demo.

## 16. Container/workspace

One reproducible Parallax runtime container does not mean one process or
one ROS node.

The image owns ROS 2 Humble, compatible Isaac ROS 3.2 components,
CUDA/TensorRT dependencies, NanoOWL/SAM dependencies, OMPL, Parallax
packages, and build tooling.

Compose owns NVIDIA runtime, host networking/IPC, `/dev/video0`, LiDAR
serial device, volumes, and ROS runtime configuration. Prefer explicit
device mappings over `privileged: true` unless demonstrated otherwise.

Suggested workspace:

``` text
ros2_ws/src/
  bringup/      # launch + config
  camera/       # SBS / calibration glue
  perception/   # NanoOWL / SAM / Object3D / tracking
  fusion/       # stereo/LiDAR association
  planning/     # EsdfCache / OMPL / semantic goals
  interfaces/   # only genuinely necessary custom interfaces
```

## 17. Aggressive build order

### Phase 0 --- Runtime skeleton

Create the v3.1 branch/workspace, image, compose setup, packages,
bringup, and configuration structure.

**Gate:** `docker compose up` produces a clean ROS graph.

### Phase 1 --- Real stereo

V4L2 → SBS → split → CameraInfo → rectify.

**Gate:** synchronized calibrated real left/right images are visible.

### Phase 2 --- Restore the RXSIM spatial spine

Real stereo → cuVSLAM + stereo depth → nvblox.

**Gate:** moving the rig produces valid pose/TF, depth, persistent
reconstruction, and configured 3D ESDF in Foxglove.

**At this point Parallax v3.1 is alive. Everything else adds capability
to a running system.**

### Phase 3 --- C1

Integrate `sllidar_ros2`.

**Gate:** LaserScan is correctly framed and coexists independently with
the spatial spine.

### Phase 4 --- Manual 3D planning

Goal → TF pose → nvblox ESDF snapshot → EsdfCache → OMPL → Path.

**Gate:** an arbitrary reachable XYZ goal from Foxglove produces a
collision-aware 3D path. Include a mapped case requiring a detour if
possible.

### Phase 5 --- NanoOWL

Port detection as a ROS-integrated capability.

**Gate:** explicit activation produces detections; shutdown can release
resources.

### Phase 6 --- EfficientViT-SAM

Integrate segmentation downstream of detections.

**Gate:** selected detections produce masks without permanently forcing
unrelated capabilities active.

### Phase 7 --- Object3D + DCF tracking

Detection + mask + stereo depth → metric Object3D → tracking.

**Gate:** world-resolved Object3D geometry appears in Foxglove; tracking
retains required neural dependencies; the final neural consumer stopping
releases neural resources.

### Phase 8 --- LiDAR evidence

Port useful C1 association/evidence behavior only.

### Phase 9 --- Semantic path

Tracked Object3D → approach point → existing planner.

**Gate:** a request such as `path to cup` resolves the object, produces
a free goal, queries ESDF, plans, publishes, and displays the path.

### Phase 10 --- Lifecycle hardening

Audit camera, stereo, cuVSLAM, nvblox, neural inference, segmentation,
tracking, LiDAR, planning, visualization, and recording for
dependency-aware resource behavior.

### Phase 11 --- Demo hardening

Tune rates/parameters, profile memory/GPU, remove unnecessary copies,
validate restart behavior, save Foxglove layout, configure rosbag2,
document commands, and remove obsolete migrated infrastructure.

## 18. Validation gates

**Camera:** stable capture, correct split, timestamps, CameraInfo,
optical frames, rectification.

**cuVSLAM:** real stereo input, valid TF, moving pose, no simulation
dependency.

**Depth:** plausible metric depth aligned to camera geometry.

**nvblox:** persistent integration, useful mesh/TSDF, updating 3D ESDF,
meaningful known-observed ESDF query.

**Neural:** explicit start/stop, actual GPU-memory release, implicit
retention while tracking.

**Object3D:** metric segmentation-supported extent, world transform,
authoritative geometry shared by 2D/3D views, persistent tracking state.

**LiDAR:** standard LaserScan, correct frame/extrinsics, independent
operation, usable metric evidence.

**Planner:** TF start, map-frame goal, populated EsdfCache, unknown
invalid, radius/margin applied, edge validation configured, expandable
AABB, `nav_msgs/Path`.

**Runtime:** one capability cannot tear down a dependency still in use;
zero-consumer expensive resources can be released; stale products can be
cleared independently; capabilities can restart cleanly.

## 19. Demo sequence

**A --- Spatial reconstruction:** start v3.1, show stereo, move rig,
show cuVSLAM trajectory and persistent nvblox reconstruction.

**B --- Manual 3D path:** publish XYZ goal, query ESDF, run OMPL, show
3D path.

**C --- Semantic path:** activate perception, identify/segment/track an
object, show metric Object3D, resolve approach point, plan through the
same planner.

**D --- Managed capabilities:** start neural perception, start DCF
tracking, disable explicit neural request and show neural remains
active, stop tracking and show resources release while
mapping/localization remain independent.

**E --- Sensor separation:** show C1 scan/evidence while nvblox remains
a stereo/depth volumetric map.

## 20. Do not build unless forced

Do not spend v3.1 time building another generic runtime/DAG, message
bus, TF system, timestamp protocol, camera-calibration protocol, QoS
layer, generic lifecycle framework, stereo algorithm, SLAM system,
TSDF/ESDF mapper, RRT implementation, raw C1 transport, visualization
transport, Nav2 3D workaround, MoveIt wrapper, controller/dynamics
layer, elaborate NLP, or persistent relocalization before the core demo
works.

Do not port code solely because it exists in `main`.

## 21. What Parallax owns

Parallax owns the complete integrated behavior. Likely retained/custom
code includes:

-   physical SBS boundary/glue where needed;
-   rig calibration integration;
-   NanoOWL and EfficientViT-SAM integration;
-   Object3D;
-   DCF/extended-object tracking behavior;
-   stereo/LiDAR evidence association;
-   semantic target resolution;
-   approach-point generation;
-   nvblox ESDF snapshot adapter;
-   local EsdfCache;
-   OMPL validity integration;
-   dependency glue not already provided by ROS;
-   minimal application-specific interfaces;
-   system bringup/configuration.

The underlying algorithms coming from multiple libraries is not a
weakness. **Integrating them into one coherent real-world system is
Parallax.**

## 22. Immediate checklist

-   [ ] Create v3.1 branch/workspace.
-   [ ] Establish ROS 2 Humble + Isaac ROS 3.2 runtime image.
-   [ ] Bring over RXSIM cuVSLAM/nvblox reference configs.
-   [ ] Remove simulation-only assumptions.
-   [ ] Publish real AR0234 SBS into ROS.
-   [ ] Split synchronized left/right.
-   [ ] Publish authoritative per-eye CameraInfo.
-   [ ] Rectify.
-   [ ] Feed real stereo into cuVSLAM.
-   [ ] Produce stereo depth.
-   [ ] Feed depth + TF into nvblox.
-   [ ] Verify persistent map + 3D ESDF.
-   [ ] Save initial Foxglove layout.
-   [ ] Integrate `sllidar_ros2`.
-   [ ] Add manual XYZ goal.
-   [ ] Add nvblox ESDF AABB snapshot.
-   [ ] Implement EsdfCache.
-   [ ] Connect OMPL R³ / RRTConnect.
-   [ ] Publish `nav_msgs/Path`.
-   [ ] Port NanoOWL.
-   [ ] Port EfficientViT-SAM.
-   [ ] Port Object3D / DCF tracking.
-   [ ] Implement dependency-aware neural resource lifetime.
-   [ ] Port LiDAR metric evidence/association.
-   [ ] Add semantic target → approach point → existing planner.
-   [ ] Audit lifecycle/resource behavior across expensive subsystems.
-   [ ] Configure rosbag2 and final Foxglove demo.

## 23. Finished-system boundary

``` text
Linux / JetPack / Docker
          |
          v
ROS 2 Humble + Isaac ROS
          |
          +-- camera/image pipeline
          +-- cuVSLAM
          +-- stereo depth
          +-- nvblox
          +-- sllidar_ros2
          |
          v
PARALLAX
          |
          +-- integrated perception
          +-- Object3D
          +-- tracking/fusion
          +-- semantic targets
          +-- capability/resource behavior
          +-- 3D planning integration
          |
          v
Foxglove / ROS interfaces
```

Parallax is not a replacement for ROS 2, Isaac ROS, nvblox, OMPL,
NanoOWL, SAM, or Foxglove. Parallax is the system produced by
integrating those capabilities around real sensors, shared geometry,
semantic object understanding, managed resources, and 3D planning.

## One-sentence goal

> **Parallax v3.1 aggressively integrates proven ROS 2, Isaac ROS,
> perception, mapping, tracking, and planning components into a managed
> real-hardware system that turns synchronized sensor observations into
> persistent spatial understanding, tracked semantic 3D objects, and
> collision-aware 3D paths.**
