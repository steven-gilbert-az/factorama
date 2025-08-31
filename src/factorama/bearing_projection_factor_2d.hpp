#pragma once
#include "factorama/types.hpp"
#include "factorama/pose_variable.hpp"
#include "factorama/landmark_variable.hpp"
#include "factorama/random_utils.hpp"
#include <memory>
#include <cassert>

namespace factorama
{
    class BearingProjectionFactor2D : public Factor
    {
    public:
        BearingProjectionFactor2D(int id,
                                  std::shared_ptr<PoseVariable> pose,
                                  std::shared_ptr<LandmarkVariable> landmark,
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

        std::vector<std::shared_ptr<Variable>> variables() override
        {
            return {pose_, landmark_};
        }

        Eigen::VectorXd compute_residual() const override;
        void compute_jacobians(std::vector<Eigen::MatrixXd>& jacobians) const override;

    private:
        int id_;
        std::shared_ptr<PoseVariable> pose_;
        std::shared_ptr<LandmarkVariable> landmark_;
        Eigen::Vector3d bearing_C_observed_;                    // measurement bearing (unit)
        Eigen::Matrix<double, 3, 2> T_;        // precomputed orthonormal basis (from k)
        double weight_;
        double reverse_depth_tolerance_;                           // small guard for alpha

        void compute_tangent_basis();          // helper to compute T from k
    };
}