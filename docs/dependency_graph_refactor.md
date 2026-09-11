# Dependency Graph Refactor

## Purpose

Parallax originally used a mostly sequential `Pipeline::process()`
architecture. That structure was appropriate while the camera, ISP,
stereo, depth, marker pose, projection, and Foxglove paths were being
brought online.

It stopped being appropriate once the runtime needed to combine neural
perception, tracking, asynchronous accelerator work, rate-mismatched
consumers, spatial association, and stateful localization.

The dependency-graph refactor replaced that orchestration model without
rewriting the algorithms that were already working.

This document records the design decisions behind the refactor and the
contracts the current runtime depends on. The implementation phases used
to get here are retained elsewhere as project history; they are not part
of the runtime model.

## The Problem With the Sequential Pipeline

The old pipeline made execution order implicit.

If a new capability needed depth, it was tempting to add another stage
after depth. If it needed detections, another stage could be added after
detection. Over time this creates one large critical path even when the
consumers have unrelated timing requirements.

That has several bad effects on an embedded perception system:

-   a slow producer can stall unrelated work;
-   expensive capabilities run even when nothing consumes them;
-   asynchronous accelerator work gets collapsed back into global
    synchronization;
-   frame provenance becomes difficult to preserve when branches run at
    different rates;
-   adding a stateful subsystem encourages more special cases in the
    central loop;
-   buffer lifetime becomes difficult to reason about because every
    stage appears to own the current frame.

The refactor moved those decisions out of the pipeline and into explicit
producer, product, dependency, and execution contracts.

## Migration, Not Rewrite

The refactor was intentionally conservative about working algorithms.

Camera acquisition did not need a new abstraction because a dependency
graph was fashionable. Stereo matching did not need to be reimplemented.
Existing VPI, CUDA, TensorRT, calibration, and visualization behavior
was kept wherever its behavior was already verified.

The work focused on orchestration.

That distinction was important because the highest-risk parts of
Parallax were already the hardware and accelerator boundaries. Rewriting
those at the same time as the execution model would have made
regressions unnecessarily difficult to isolate.

The rule throughout the refactor was to move one verified capability at
a time behind a clearer contract and keep the repository buildable.

Existing comments that explained hardware behavior, calibration
assumptions, memory ownership, or non-obvious implementation decisions
were also treated as part of the engineering record rather than
disposable cleanup.

## Product Contracts

The graph is built around typed products.

A product is more than a pointer to the latest result. Its contract
includes the information needed for another producer to decide whether
that result is valid for its own work.

Depending on the product, that includes:

-   source sequence and timestamp;
-   producer identity;
-   image or coordinate space;
-   validity;
-   source provenance;
-   memory domain;
-   generation or revision identity;
-   compatibility information.

This became especially important once asynchronous branches were
introduced.

A fresh detection and a fresh depth map are not automatically
compatible. If they were derived from unrelated camera observations,
combining them can create a plausible-looking but incorrect 3D result.

The product contract therefore preserves source identity through the
graph rather than reducing everything to "latest."

## Producer Contracts

A producer declares what it consumes and what it publishes.

The runtime can then resolve dependencies from the requested output
instead of relying on a manually maintained call sequence.

The producer boundary also makes execution behavior explicit. A producer
may be stateless or stateful, newest-value or ordered, CPU-bound or
accelerator-backed, cheap enough to run continuously or expensive enough
to activate only when demanded.

Those differences belong in the execution contract because they affect
correctness, not just performance.

A producer should not reach through the runtime to obtain arbitrary
resources. `ExecutionContext` is intentionally constrained so that
dependencies remain visible and resource ownership does not disappear
behind a global service locator.

## Demand

Demand comes from application behavior and active consumers.

A capability can be registered without running. A Foxglove topic can
exist without being continuously serialized. A detector can be available
without executing until a command or dependent product requires it.

The resolver computes the required producer subgraph from the requested
products.

This lets Parallax keep the graph static enough to reason about while
making actual execution dynamic.

It also avoids a common failure mode in modular perception systems where
"modular" means every module is running all the time and publishing into
a large message bus whether or not its outputs are useful.

## Newest Useful Data

Most realtime edges use newest-useful-data semantics.

For a stateless consumer, processing an old camera frame simply because
it exists is usually worse than using the newest compatible frame.
Ordered history increases latency and retains buffers that may be large
or accelerator-backed.

For those paths, stale pending work may be dropped or superseded
deliberately.

This behavior is not universal.

A producer that depends on temporal continuity must say so. cuVSLAM is
the clearest current example: its synchronized stereo input is ordered
estimator history, not a queue of interchangeable frames.

The graph therefore treats ordered history as an explicit requirement
rather than the default behavior for every edge.

## Product Store and Bounded Lifetime

The product store supports both latest-value access and bounded ordered
history.

It is not an unbounded broker.

This matters on the Orin Nano because shared ownership does not make
large buffers free. A product may be logically lightweight while still
keeping a camera image, VPI allocation, or other backing storage alive
through a reference chain.

History is bounded, stale work can be superseded where semantically
safe, and consumers are expected to release products when they are no
longer useful.

The goal is predictable lifetime rather than relying on eventual
reference-count cleanup under memory pressure.

## Readiness, Compatibility, and Failure

One of the more important runtime corrections was separating readiness
from failure.

An asynchronous producer frequently has no valid work for a scheduling
pass. Its compatible input may not have arrived yet, the current
observation may already have been superseded, or downstream demand may
have disappeared.

Those conditions are normal.

The runtime therefore distinguishes:

``` text
NoWork
```

from:

``` text
Failed
```

`NoWork` means the producer did not have a valid execution to perform.
`Failed` means the producer attempted real work and that work failed.

The same separation exists between freshness and source compatibility. A
product can be recent but incompatible, or compatible but too old for a
particular consumer.

Treating those as separate concepts made asynchronous and
rate-mismatched branches substantially easier to reason about.

## Accelerator Residency

The refactor preserves accelerator residency instead of normalizing
every product through CPU memory.

VPI, CUDA, TensorRT, and cuVSLAM already have their own execution and
memory models. The Parallax graph coordinates dependencies around those
systems; it does not replace them.

Synchronization is therefore local to actual dependency boundaries.

A producer should wait for the event or resource it genuinely depends
on, not for a global "frame complete" barrier that serializes unrelated
branches.

This is particularly important for VPI work because an apparently
convenient synchronization call can erase the concurrency the graph was
introduced to preserve.

## Stateful Producers

Stateful producers require more care than stateless image transforms.

DCF maintains a selected target across observations. cuVSLAM maintains
an estimator whose state depends on ordered stereo input. Their outputs
can still participate in the normal product graph, but their internal
continuity cannot be inferred from latest-value storage.

The refactor therefore does not assume that a `stateful` flag alone
creates correct sequencing.

The producer or its ingress path must own the ordering semantics it
requires.

This keeps the generic scheduler relatively small and prevents it from
becoming a second implementation of the state machines already owned by
the underlying algorithms.

## Spatial Association as a Consumer

A useful result of the refactor is that semantic perception and metric
geometry no longer need to be fused inside a tracker.

NanoOWL can produce a semantic detection. Stereo can independently
produce depth. Spatial association can consume compatible observations
from both branches and publish an `Object3D`.

A DCF target can feed the same association path when persistent target
state is required.

This removed the earlier assumption that multi-object tracking had to
sit between detection and 3D perception. Persistent identity is useful
when the application asks for it, but it is not a prerequisite for
estimating where a detected object is in space.

That change made the product model simpler and better matched the actual
capabilities of the system.

## Localization as the Ordered Exception

cuVSLAM exposed the limit of treating every realtime edge as
newest-value.

Visual localization is stateful and temporally ordered. Skipping from
one arbitrary latest stereo observation to another changes the estimator
input history and can invalidate the session.

Localization therefore uses bounded ordered ingress for synchronized
stereo observations while publishing its outputs back into the normal
product store.

This is a useful example of why the graph has execution policies rather
than one universal queueing rule.

Demand can still control downstream localization products and
publication. It should not silently destroy estimator continuity because
a visualization panel was closed.

## Commands and Publication

Commands request outcomes rather than directly invoking internal stages.

This allows the same dependency graph to serve interactive requests
without making the command parser part of the perception implementation.

Publication follows the same rule. Foxglove topics are declared
statically, but active subscriptions determine what needs to be
serialized and sent.

The distinction between computation and publication is intentional. A
product may need to exist for another producer even when nobody is
currently visualizing it.

Likewise, an estimator may need to continue running even when its
visualization topic has no subscribers.

## Observability

The refactor also changed what needs to be measured.

End-to-end FPS and latency are useful, but they are not enough to
diagnose a heterogeneous graph.

The runtime needs visibility into:

-   producer executions;
-   skipped, dropped, and superseded work;
-   queue/history pressure;
-   allocations;
-   memory-domain transfers;
-   synchronization;
-   producer latency and rate.

These measurements make it possible to tell the difference between an
algorithm that is expensive, a scheduler that is doing unnecessary work,
a buffer lifetime problem, and an accelerator boundary that is forcing
copies or synchronization.

## Result

The dependency-graph refactor is complete through the current
perception, spatial association, command, observability, and cuVSLAM
localization work.

The practical result is not a novel graph framework. It is a runtime
structure that fits the actual workload Parallax has accumulated.

Stereo geometry can run independently of neural perception. Detection
does not require segmentation. A one-shot detection does not require
multi-object tracking. DCF can maintain a selected target when
persistence is useful. Spatial association can combine semantic and
metric observations without owning either producer. cuVSLAM can preserve
ordered estimator state without forcing every other edge to become
ordered. Foxglove can observe the system without defining its internal
lifecycle.

That is the reason for the refactor: each subsystem keeps the execution
semantics it actually needs while still participating in one explicit
dependency model.
