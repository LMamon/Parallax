#pragma once

#include <memory>

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>

#include "planning/esdf_grid.hpp"

namespace planning {

class EsdfMotionValidator : public ompl::base::MotionValidator {
    public:
        EsdfMotionValidator(const ompl::base::SpaceInformationPtr &si,
                            std::shared_ptr<const EsdfGrid> grid,
                            double clearance_m,
                            double check_resolution_m);

        bool checkMotion(const ompl::base::State *s1, const ompl::base::State *s2) const override;

        bool checkMotion(const ompl::base::State *s1,
                         const ompl::base::State *s2,
                         std::pair<ompl::base::State *, double> &lastValid) const override;

    private:
        bool stateValid(const ompl::base::State *state) const;

        std::shared_ptr<const EsdfGrid> grid_;
        double clearance_m_;
        double check_resolution_m_;
    };

}  // namespace planning
