#pragma once
#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/perception/stereo_roi_associator.hpp>
#include <vector>
namespace parallax::perception {
class TrackedObject3DProducer final : public core::Producer {
 public:
  TrackedObject3DProducer(StereoRoiAssociator& stereo, core::ProductStore& products) noexcept;
  [[nodiscard]] std::string_view name() const noexcept override;
  [[nodiscard]] const std::vector<core::ProductId>& inputs() const noexcept override;
  [[nodiscard]] const std::vector<core::ProductId>& outputs() const noexcept override;
  [[nodiscard]] const std::vector<core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;
  [[nodiscard]] core::ExecutionPolicy execution_policy() const noexcept override;
  core::SubmitResult submit(core::ExecutionContext& context) override;
 private:
  static constexpr std::size_t DepthHistoryCapacity=4;
  StereoRoiAssociator& stereo_; core::ProductStore& products_; core::SourceObservation last_track_observation_{};
  const std::vector<core::ProductId> inputs_{core::ProductId::Track2D,core::ProductId::Depth};
  const std::vector<core::ProductId> outputs_{core::ProductId::TrackedObject3D};
  const std::vector<core::CompatibleInputRequirement> compatible_inputs_{{core::ProductId::Depth,DepthHistoryCapacity}};
};
}
