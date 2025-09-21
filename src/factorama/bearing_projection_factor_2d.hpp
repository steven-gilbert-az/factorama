#pragma once
#include "factorama/types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include "factorama/random_utils.hpp"
#include <memory>
#include <cassert>

namespace factorama
{
    /**
     * @brief 2D bearing projection factor for camera measurements
     *
     * Projects bearing differences onto the tangent plane of the observed bearing direction.
     * Provides a 2D residual instead of the full 3D residual from BearingObservationFactor.
     *
     * @code
     * Eigen::Vector3d bearing_vector(0.0, 0.0, 1.0);
     * auto bearing_factor = std::make_shared<BearingProjectionFactor2D>(
     *     factor_id++, camera_pose, landmark, bearing_vector, sigma);
     * @endcode
     */
    class BearingProjectionFactor2D : public Factor
    {
    public:
        /**
         * @brief Construct 2D bearing projection factor
         * @param id Unique factor identifier
         * @param pose Camera pose variable
         * @param landmark 3D landmark variable
         * @param bearing_C_observed Unit bearing vector in camera frame
         * @param sigma Standard deviation of angular measurement (radians)
         * @param along_tolerance_epsilon Tolerance for numerical stability
         */
        BearingProjectionFactor2D(int id,
                                  PoseVariable* pose,
                                  LandmarkVariable* landmark,
                                  const Eigen::Vector3d& bearing_C_observed,
                                  double sigma = 1.0,
                                  double along_tolerance_epsilon = 1e-6)
            : id_(id),
              pose_(pose),
              landmark_(landmark),
              bearing_C_observed_(bearing_C_observed.normalized()),
              weight_(1.0 / sigma),
              reverse_depth_tolerance_(along_tolerance_epsilon)
        {
            assert(pose != nullptr && "pose cannot be nullptr");
            assert(landmark != nullptr && "landmark cannot be nullptr");
            assert(sigma > 0.0 && "Sigma must be greater than zero");
            compute_tangent_basis();
        }

        int id() const override { return id_; }
        
        int residual_size() const override { return 2; }
        
        double weight() const override { return weight_; }
        
        std::string name() const override 
        { 
            return "BearingProjection2D(" + pose_->name() + "," + landmark_->name() + ")";
        }
        
        FactorType::FactorTypeEnum type() const override 
        { 
            return FactorType::bearing_observation; // Using existing enum for now
        }

        std::vector<Variable *> variables() override
        {
            return {pose_, landmark_};
        }

        Eigen::VectorXd compute_residual() const override;
        void compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const override;

    private:
        int id_;
        PoseVariable* pose_;
        LandmarkVariable* landmark_;
        Eigen::Vector3d bearing_C_observed_;                    // measurement bearing (unit)
        Eigen::Matrix<double, 3, 2> T_;        // precomputed orthonormal basis (from k)
        double weight_;
        double reverse_depth_tolerance_;                           // small guard for alpha

        void compute_tangent_basis();          // helper to compute T from k
    };
}