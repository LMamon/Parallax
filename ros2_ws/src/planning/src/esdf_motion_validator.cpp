#include "planning/esdf_motion_validator.hpp"

#include <algorithm>
#include <cmath>

#include <ompl/base/spaces/RealVectorStateSpace.h>

namespace planning {

    EsdfMotionValidator::EsdfMotionValidator(const ompl::base::SpaceInformationPtr &si,
                                            std::shared_ptr<const EsdfGrid> grid,
                                            const double clearance_m,
                                            const double check_resolution_m)
                                            : ompl::base::MotionValidator(si),
                                            grid_(std::move(grid)),
                                            clearance_m_(clearance_m),
                                            check_resolution_m_(check_resolution_m) {}

    bool EsdfMotionValidator::stateValid(const ompl::base::State *state) const {

        const auto *xyz = state->as<ompl::base::RealVectorStateSpace::StateType>();

        return grid_->isStateValid(xyz->values[0],
                                xyz->values[1],
                                xyz->values[2],
                                clearance_m_);
    }

    bool EsdfMotionValidator::checkMotion(const ompl::base::State *s1, const ompl::base::State *s2) const {

        std::pair<ompl::base::State *, double> unused{nullptr, 0.0};
        return checkMotion(s1, s2, unused);
    }

    bool EsdfMotionValidator::checkMotion(const ompl::base::State *s1,
                                        const ompl::base::State *s2,
                                        std::pair<ompl::base::State *, double> &lastValid) const {

        if (!stateValid(s1)) {
            lastValid.second = 0.0;
            return false;
        }

        const double distance = si_->distance(s1, s2);

        const unsigned int segments = std::max(1u,
                                    static_cast<unsigned int>(std::ceil(distance / check_resolution_m_)));

        auto test_state = si_->allocState();

        for (unsigned int i = 1; i <= segments; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(segments);

            si_->getStateSpace()->interpolate(s1, s2, t, test_state);

            if (!stateValid(test_state)) {
                lastValid.second = static_cast<double>(i - 1) / static_cast<double>(segments);

                if (lastValid.first != nullptr) {
                    si_->getStateSpace()->interpolate(s1,
                                                    s2,
                                                    lastValid.second,
                                                    lastValid.first);
                }

                si_->freeState(test_state);
                return false;
            }
        }

        si_->freeState(test_state);

        lastValid.second = 1.0;

        if (lastValid.first != nullptr) {
            si_->copyState(lastValid.first, s2);
        }

        return true;
    }

}  // namespace planning
