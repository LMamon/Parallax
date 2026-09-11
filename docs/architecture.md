## Overview

Parallax is a perception and localization runtime for an unmanned system
built around the NVIDIA Jetson Orin Nano. It combines a the following in one runtime:

- calibrated stereo camera
- 2D LiDAR
- hardware-accelerated image processing
- neural perception
- target tracking
- metric spatial association
- visual localization
- Foxglove observability

The important architectural decision is that these capabilities are not
treated as stages in one large frame loop. They are represented as
products with explicit dependencies. Producers run when their outputs
are required, and independent branches are allowed to run at rates that
make sense for the work they perform.

The result is a heterogeneous dependency graph rather than a traditional
sequential perception pipeline.

The basic rule is simple:

> Products are declared statically. Computation and publication are
> demand-driven.

This matters on an embedded system because not every useful product
needs to exist at camera rate, not every consumer needs the same frame,
and not every operation belongs on the same processor or accelerator.

## Why the Runtime Uses a Dependency Graph

The original implementation was a conventional pipeline. Camera
acquisition was followed by ISP, rectification, stereo matching, depth,
pose estimation, visualization, and whatever else had been added to the
loop.

That was useful while bringing the hardware and algorithms online
because the execution order was obvious and easy to debug. It became a
poor fit once Parallax started combining workloads with different timing
and state requirements.

Stereo depth may be useful at a high rate. A neural detector may run
more slowly. Segmentation may only be needed after a detection request.
A tracker is stateful and needs continuity. Localization must consume
synchronized stereo observations in order. Foxglove may subscribe to any
combination of those products without being responsible for how they are
computed.

Putting all of that in one sequential loop either wastes compute or
forces unrelated work to wait on the slowest stage.

The dependency graph separates those concerns. Each producer declares
what it consumes and what it produces. The runtime resolves the
dependencies required for the current demand and schedules the resulting
work according to its execution contract.

This is not intended to be a general-purpose scheduler. VPI, CUDA,
TensorRT, cuVSLAM, and the CPU runtime already provide the mechanisms
used to execute the actual work. Parallax is responsible for deciding
what work is required, whether its inputs are compatible, and how the
resulting products move through the system.

## Products and Producers

A product is a typed result that can be consumed by another part of the
graph.

Examples include camera frames, rectified stereo images, disparity,
confidence, depth, detections, segmentation masks, target state, 3D
objects, localization poses, visual observations, landmarks, and
trajectory history.

A producer owns the computation that creates one or more products. Its
interface declares its inputs, outputs, and execution behavior instead
of relying on an implicit call order in a central pipeline.

This gives the graph a useful property: adding a consumer does not
require moving the algorithm that produces its data into the consumer.

Depth is a good example. Stereo depth exists as a product because
several unrelated consumers may need it. A visualization can display it,
a marker projection can query it, and a semantic observation can use it
to estimate an object's position. None of those consumers owns stereo
processing.

The same rule applies to neural perception. NanoOWL produces semantic
detections. EfficientViT-SAM can refine a requested observation with
segmentation. DCF can maintain a selected target through time. Those
products can then be combined with metric observations without turning
detection, segmentation, tracking, and depth into one inseparable
subsystem.

## Product Store

The product store is the boundary between producers and consumers.

Most realtime products use newest-useful-data semantics. If a consumer
only needs the current depth map or current detection result, keeping a
large ordered queue of old products adds memory pressure and latency
without adding value.

Ordered history is therefore opt-in.

This distinction is important because some consumers genuinely require
sequence continuity. cuVSLAM is the clearest example. Localization
cannot be treated as a newest-frame-only consumer because its estimator
state depends on synchronized stereo observations arriving in
monotonically ordered time.

The store therefore supports two different ideas without conflating
them:

-   latest-value products for normal realtime consumption;
-   bounded ordered history for consumers that require temporal
    continuity.

Both are bounded. Large image and accelerator-backed buffers cannot
accumulate indefinitely simply because a product uses shared ownership.

Products retain timestamps, sequence identity, producer provenance, and
the metadata needed to determine whether two observations are actually
compatible.

## Freshness and Compatibility

Freshness and compatibility are separate questions.

A product can be recent and still be the wrong source observation for
another product. A detection from one camera observation should not be
associated with an unrelated depth frame simply because both happened
recently.

For branches that converge, Parallax preserves enough provenance to
determine whether inputs refer to compatible source observations.
Freshness policies can then answer a different question: whether an
otherwise compatible product is still useful to the consumer.

This distinction became necessary once neural perception, depth,
tracking, and localization began operating asynchronously and at
different rates.

The runtime should never manufacture synchronization by silently pairing
whatever values happen to be newest.

## Execution Policies

Different producers have different execution requirements, so scheduling
behavior is part of the producer contract.

Policies describe concerns such as target rate, priority, freshness
limits, drop behavior, statefulness, resource affinity, and whether a
producer consumes latest values or ordered history.

The default realtime behavior is to prefer the newest useful work. If
several camera observations arrive while a stateless consumer is busy,
processing every stale intermediate observation is usually the wrong
tradeoff.

Stateful estimators and trackers are different. Their continuity
requirements have to be represented explicitly rather than assumed from
a generic producer interface.

A producer that has no compatible work is not considered failed. There
is a meaningful difference between "nothing should run for this
observation" and "the producer attempted work and failed." Keeping that
distinction prevents normal asynchronous behavior from being reported as
an error condition.

## Accelerator Ownership and Synchronization

Parallax is designed around the accelerators already available on the
Jetson platform.

Image processing and stereo work use VPI and the hardware paths it
exposes. Neural inference uses TensorRT and CUDA. Localization uses
NVIDIA cuVSLAM. CPU work remains on CPU workers where that is the
appropriate execution domain.

The graph does not force intermediate data back through CPU memory
merely to make ownership simpler.

Accelerator-resident products remain resident where practical, and
synchronization is introduced at real dependency boundaries rather than
globally after every stage. VPI streams and events are used to preserve
asynchronous execution where the underlying operations allow it.

This is one of the reasons `ExecutionContext` is intentionally narrow.
It provides the resources a producer is supposed to use without becoming
a service locator that exposes every runtime object to every subsystem.

## Image and Stereo Geometry

The image branch starts with synchronized stereo acquisition and
produces the calibrated image products used by the rest of the system.

The existing camera path includes image signal processing, stereo
rectification, disparity, confidence, and metric depth. The stereo
geometry is calibrated and shared rather than redefined by downstream
consumers.

The camera and LiDAR mounting transforms are also treated as existing
sensor geometry. Consumers use the configured transforms to move
observations between sensor and body frames instead of introducing
subsystem-specific coordinate conventions.

This keeps geometric reasoning centralized and makes later products
easier to interpret.

## Neural Perception and Target Tracking

Open-vocabulary detection is provided by NanoOWL. A detection contains
semantic information and image-space geometry, but it is not treated as
a complete 3D object.

EfficientViT-SAM provides segmentation when a more precise
image-supported region is useful. It is not required to run for every
detection.

Single-target DCF tracking is used when Parallax is asked to maintain a
selected target through time. Tracking is intentionally separate from
detection. A one-shot semantic detection can participate in spatial
association without first being promoted into a persistent track.

This was an important simplification. Multi-object tracking is not a
prerequisite for producing useful spatial observations.

## Spatial Association

Spatial association combines semantic image observations with metric
sensor information.

A NanoOWL detection can be associated directly with stereo depth to
produce an `Object3D`. A DCF-maintained target can use the same
association path while retaining its persistent target identity.

Segmentation-supported depth can provide better image support when it is
available. A bounding-box observation can still use a smaller valid
depth region when segmentation is unnecessary.

The 2D LiDAR is also available as a metric source when its scan plane
actually intersects the observed object. It is not treated as a general
replacement for stereo depth because the LiDAR only measures one
horizontal slice of the environment.

The resulting 3D representation is deliberately conservative. If
Parallax only has a defensible XYZ estimate, it publishes a point-like
spatial observation. It does not turn a 2D bounding box into a
metrically authoritative 3D cuboid without evidence for the object's
physical extent.

This keeps the product semantics aligned with what the sensors actually
measured.

## Localization

Localization is provided by NVIDIA cuVSLAM using synchronized rectified
stereo observations.

Unlike most producers in the graph, cuVSLAM is a continuously ordered
estimator. Its input cannot be freely superseded by newer observations
without changing the meaning of the estimator state.

For that reason, localization uses bounded ordered stereo ingress while
its downstream products continue to participate in the normal product
system.

The localization branch currently exposes the current pose, bounded
trajectory history, tracking state, visual observations, and visual
landmarks. These products are expressed in a localization world frame so
that sensor motion and spatial observations can be viewed in a common
coordinate system.

Semantic spatial observations can also be transformed into this world
frame. This is the point where perception and localization become one
coherent spatial system: the camera pose changes as the platform moves
while an observed stationary object remains approximately stationary in
the world frame.

Localization continuity is not owned by Foxglove. A visualization
subscription may determine whether a product is serialized and
published, but the estimator lifecycle is an application/runtime
concern.

## Foxglove and Commands

Foxglove is the primary observability and command surface for Parallax.

Topics are declared as part of the system interface, but declaration
does not imply continuous computation or publication. Subscriber demand
can activate visualization products without turning Foxglove into the
owner of the underlying algorithm.

Commands represent requested behavior rather than direct calls into
implementation stages. A command such as `detect cup` can activate
semantic detection and its required dependencies. `track cup` can
establish a persistent DCF target. Existing geometric commands can
request depth or marker-related products without being routed through a
neural detector.

This keeps the command interface aligned with capabilities instead of
implementation details.

Foxglove is also where the coordinate-frame model becomes useful. Stereo
geometry, LiDAR, current localization pose, trajectory, landmarks, and
semantic 3D observations can be inspected in the same 3D environment
without maintaining a second visualization application.

## Observability

FPS by itself is not enough to understand a heterogeneous runtime.

Parallax records execution behavior at the producer level, including
producer executions, dropped or superseded work, memory-domain
transfers, allocations, and synchronization. Latency and throughput
still matter, but they do not explain why a graph is slow or where
embedded memory pressure is coming from.

The goal is to make scheduling and memory behavior observable enough
that performance decisions can be based on the runtime that actually
exists rather than on assumptions about a nominal camera frame rate.

## Current System Boundary

Parallax currently provides the perception and localization part of an
autonomous or remotely operated unmanned system. 

It acquires and processes stereo imagery, produces metric depth,
performs semantic detection and segmentation, maintains a selected
visual target, associates semantic observations with 3D position,
consumes 2D LiDAR data, estimates visual localization, and exposes those
products through a common coordinate-frame and observability model.

It does not currently include vehicle actuation, flight control, or a
navigation planner. Those systems can consume Parallax products later
without requiring the perception runtime to become a vehicle controller.

That boundary is intentional. Parallax is responsible for producing the 
coherent visual system of the platform and its environment. Higher-level
coordination can decide what to do with them.
