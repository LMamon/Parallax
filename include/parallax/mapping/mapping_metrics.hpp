#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
namespace parallax::mapping {
struct StageMetrics {
 std::atomic<std::uint64_t> samples{0}, total_us{0}, last_us{0}, max_us{0};
 void observe(std::uint64_t us) noexcept { samples.fetch_add(1); total_us.fetch_add(us); last_us.store(us); auto m=max_us.load(); while(m<us&&!max_us.compare_exchange_weak(m,us)){} }
};
struct MappingMetrics {
 std::atomic_bool profiling_sync{false};
 StageMetrics depth_integration,tsdf_snapshot,color_integration,mesh_update_and_flatten,mesh_snapshot,tsdf_publication,mesh_publication;
 std::atomic<std::uint64_t> tsdf_d2h_transfers{0},tsdf_d2h_bytes{0},mesh_materialized_host_bytes{0},map_allocated_blocks{0},map_peak_allocated_blocks{0};
 void observeBlocks(std::uint64_t n) noexcept { map_allocated_blocks.store(n); auto p=map_peak_allocated_blocks.load(); while(p<n&&!map_peak_allocated_blocks.compare_exchange_weak(p,n)){} }
};
inline MappingMetrics& mapping_metrics() noexcept { static MappingMetrics m; return m; }
class ScopedStageTimer { public: explicit ScopedStageTimer(StageMetrics& m):m_(m),t_(std::chrono::steady_clock::now()){} ~ScopedStageTimer(){auto u=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t_).count();m_.observe(u>0?static_cast<std::uint64_t>(u):0);} private: StageMetrics& m_; std::chrono::steady_clock::time_point t_;};
}
