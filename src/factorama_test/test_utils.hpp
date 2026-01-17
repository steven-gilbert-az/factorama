
#include <random>
#include <chrono>
#include "factorama/factor_graph.hpp"
#include "factorama/generic_prior_factor.hpp"
#include "factorama/pose_prior_factors.hpp"
#include "factorama/landmark_variable.hpp"
#include "factorama/bearing_observation_factor.hpp"
#include "factorama/inverse_range_variable.hpp"
#include "factorama/inverse_range_bearing_factor.hpp"
#include "factorama/pose_between_factors.hpp"
#include "factorama/bearing_projection_factor_2d.hpp"

namespace factorama
{

    constexpr double precision_tol = 1e-9;

    bool is_approx_equal(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tol = 1e-9)
    {
        if (a.rows() != b.rows()) {
            std::cout << "rows did not match: " << a.rows() << " vs " << b.rows() << std::endl;
            return false;
        }

        if (a.cols() != b.cols()) {
            std::cout << "cols did not match: " << a.cols() << " vs " << b.cols() << std::endl;
            return false;
        }
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                double diff = fabs(a(i, j) - b(i, j));
                if (diff > tol) {
                    std::cout << "diff: " << diff << ", tol: " << tol << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    inline Eigen::Vector3d random_vector3d(std::normal_distribution<double>& noise, std::mt19937& rng)
    {
        Eigen::Vector3d out;
        for (int i = 0; i < 3; i++) {
            out[i] = noise(rng);
        }
        return out;
    }

    inline void CreateSimpleScenario(std::vector<Eigen::Matrix<double, 6, 1>>& camera_poses,
                                     std::vector<Eigen::Vector3d>& landmark_positions)
    {

        camera_poses = {{-1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

        landmark_positions = {{-1.0, -1.0, 5.0}, {1.0, -1.0, 5.0}, {-1.0, 1.0, 5.0}, {1.0, 1.0, 5.0}};
    }

    inline void CreatePlanarScenario(std::vector<Eigen::Matrix<double, 6, 1>>& camera_poses,
                                     std::vector<Eigen::Vector3d>& landmark_positions)
    {

        // X - forward, Y - right, Z - down. Camera flies along "forward" direction. is slightly "above" terrain

        camera_poses = {
            {0.0, 0.0, -1.0, 0.0, 0.0, 0.0},   {1.0, 0.05, -1.0, 0.1, 0.0, 0.0}, {2.0, -0.05, -1.0, 0.0, 0.05, 0.0},
            {3.00, 0.0, -1.05, 0.0, 0.0, 0.0}, {4.0, 0.0, -1.0, 0.0, 0.0, 0.03},
        };

        // Landmarks can form a 3x3 grid
        landmark_positions = {{10.0, -2.0, 0.1}, {10.0, 0.0, 0.0},  {10.0, 2.0, -0.1},
                              {13.0, -3.0, 0.0}, {13.0, 0.0, -0.1}, {13.0, 3.0, 0.1},
                              {17.0, -4.0, 0.1}, {17.0, -1.0, 0.2}, {17.0, 4.0, 0.1}};
    }

    inline std::vector<Eigen::Vector3d> CreateLandmarksInVolume(Eigen::Vector3d min, Eigen::Vector3d max,
                                                                int num_landmarks, int seed = 42)
    {
        std::vector<Eigen::Vector3d> landmarks;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist_x(min.x(), max.x());
        std::uniform_real_distribution<double> dist_y(min.y(), max.y());
        std::uniform_real_distribution<double> dist_z(min.z(), max.z());

        for (int i = 0; i < num_landmarks; ++i) {
            Eigen::Vector3d lm;
            lm.x() = dist_x(rng);
            lm.y() = dist_y(rng);
            lm.z() = dist_z(rng);
            landmarks.push_back(lm);
        }
        return landmarks;
    }

    // Shared ground truth graph creation
    FactorGraph CreateSimpleGroundTruthGraph()
    {
        FactorGraph graph;

        int var_num = 0;
        int factor_num = 0;

        // Two camera poses at x = -1 and x = +1, looking down Z
        Eigen::Matrix<double, 6, 1> pose_left_vec = Eigen::Matrix<double, 6, 1>::Zero();
        pose_left_vec(0) = -1.0;
        auto pose_left = std::make_shared<PoseVariable>(var_num++, pose_left_vec);

        Eigen::Matrix<double, 6, 1> pose_right_vec = Eigen::Matrix<double, 6, 1>::Zero();
        pose_right_vec(0) = 1.0;
        auto pose_right = std::make_shared<PoseVariable>(var_num++, pose_right_vec);

        graph.add_variable(pose_left);
        graph.add_variable(pose_right);

        // Landmarks at (±1, ±1, 5)
        std::vector<Eigen::Vector3d> landmark_positions = {
            {-1.0, -1.0, 5.0}, {1.0, -1.0, 5.0}, {-1.0, 1.0, 5.0}, {1.0, 1.0, 5.0}};

        for (int i = 0; i < 4; ++i) {
            auto lm = std::make_shared<LandmarkVariable>(var_num++, landmark_positions[i]);
            graph.add_variable(lm);

            for (int j = 0; j < 2; ++j) {
                auto cam = (j == 0) ? pose_left : pose_right;

                // Project into camera
                Eigen::Vector3d p_C = cam->dcm_CW() * (landmark_positions[i] - cam->pos_W());
                Eigen::Vector3d bearing = p_C.normalized();

                auto factor =
                    std::make_shared<BearingObservationFactor>(factor_num++, cam.get(), lm.get(), bearing, 1.0);
                graph.add_factor(factor);
            }
        }

        return graph;
    }

    //
    inline FactorGraph CreateGraphWithLandmarks(std::vector<Eigen::Matrix<double, 6, 1>> camera_poses,
                                                std::vector<Eigen::Vector3d> landmark_positions, bool random_noise,
                                                bool constant_pose, bool prior_factors, double noise_sigma = 0.04,
                                                double sparsity = 0.0)
    {
        FactorGraph graph;
        int var_id = 0;
        int factor_id = 0;

        std::mt19937 rng(42);                                     // fixed seed for reproducibility
        std::normal_distribution<double> noise(0.0, noise_sigma); // small perturbation

        std::vector<std::shared_ptr<PoseVariable>> poses;
        std::vector<std::shared_ptr<PoseVariable>> truth_poses;
        for (size_t i = 0; i < camera_poses.size(); i++) {
            auto original_cam_pose = camera_poses[i];
            auto cam_pose = camera_poses[i];

            if (random_noise && i != 0) {
                // Noise up the camera position (except the first one)
                cam_pose.segment<3>(0) += random_vector3d(noise, rng);
                // Noise up the camera orientation
                cam_pose.segment<3>(3) += random_vector3d(noise, rng);
            }

            auto pose = std::make_shared<PoseVariable>(var_id++, cam_pose);

            if (constant_pose || i == 0) {
                pose->set_constant(true);
            } else if (prior_factors) {
                // Create a pair of prior factors (position / orientation)
                Eigen::Vector3d pos_prior = original_cam_pose.segment<3>(0);
                Eigen::Vector3d rot_prior = original_cam_pose.segment<3>(3);
                Eigen::Matrix3d dcm_prior = ExpMapSO3(rot_prior);

                auto position_prior =
                    std::make_shared<PosePositionPriorFactor>(factor_id++, pose.get(), pos_prior, noise_sigma);
                auto rotation_prior =
                    std::make_shared<PoseOrientationPriorFactor>(factor_id++, pose.get(), dcm_prior, noise_sigma);
                graph.add_factor(position_prior);
                graph.add_factor(rotation_prior);
            }
            graph.add_variable(pose);

            poses.push_back(pose);

            // add truth pose

            auto truth_pose = std::make_shared<PoseVariable>(var_id++, original_cam_pose);
            truth_poses.push_back(truth_pose);
        }
        std::vector<std::shared_ptr<LandmarkVariable>> landmarks;

        for (auto lm_pos : landmark_positions) {

            auto original_lm_pos = lm_pos;
            if (random_noise) {
                lm_pos += random_vector3d(noise, rng);
            }
            auto landmark = std::make_shared<LandmarkVariable>(var_id++, lm_pos);
            landmarks.push_back(landmark);
            graph.add_variable(landmark);
            if (prior_factors) {
                // Create a pair of prior factors (position / orientation)
                auto landmark_prior =
                    std::make_shared<GenericPriorFactor>(factor_id++, landmark.get(), original_lm_pos, noise_sigma);
                graph.add_factor(landmark_prior);
            }
        }

        for (size_t pose_num = 0; pose_num < poses.size(); pose_num++) {

            // Generate bearing measurements for each landmark
            auto pose = poses[pose_num];
            auto truth_pose = truth_poses[pose_num];
            auto dcm_CW = truth_pose->dcm_CW();
            for (size_t lm_num = 0; lm_num < landmarks.size(); lm_num++) {
                // Randomly choose whether to populate the bearing factor based on "filled_in"
                double tmp = rng();
                tmp /= static_cast<double>(rng.max());
                if (tmp < sparsity) {
                    continue; // Skip this landmark for this camera
                }

                auto lm_pos = landmark_positions[lm_num];
                auto lm = landmarks[lm_num];

                Eigen::Vector3d pos_C = dcm_CW * (lm_pos - truth_pose->pos_W());
                Eigen::Vector3d bearing = pos_C.normalized();

                double angle_noise_mult = 0.1;
                double angle_noise_sigma =
                    noise_sigma * angle_noise_mult; // assume angles sensor is more accurate than position.
                if (random_noise) {
                    bearing += random_vector3d(noise, rng) * angle_noise_mult; // Bearing noise is
                    bearing = bearing.normalized();
                }

                auto factor = std::make_shared<BearingObservationFactor>(factor_id++, pose.get(), lm.get(), bearing,
                                                                         angle_noise_sigma);
                graph.add_factor(factor);
            }
        }

        return graph;
    }

    inline FactorGraph CreateGraphWithInverseRangeVariables(std::vector<Eigen::Matrix<double, 6, 1>> camera_poses,
                                                            std::vector<Eigen::Vector3d> landmark_positions,
                                                            bool random_noise, bool constant_pose, bool prior_factors,
                                                            double noise_sigma = 0.04, double initial_range_est = -1.0)
    {
        FactorGraph graph;
        int var_id = 0;
        int factor_id = 0;

        std::mt19937 rng(42);                                     // fixed seed for reproducibility
        std::normal_distribution<double> noise(0.0, noise_sigma); // small perturbation

        std::vector<std::shared_ptr<PoseVariable>> poses;
        std::vector<std::shared_ptr<PosePositionPriorFactor>> pose_position_priors;
        std::vector<std::shared_ptr<PosePositionPriorFactor>> pose_orientation_priors;
        for (size_t i = 0; i < camera_poses.size(); i++) {
            auto& cam_pose = camera_poses[i];

            if (random_noise && i != 0) {
                // Noise up the camera position (except the first one)
                cam_pose.segment<3>(0) += random_vector3d(noise, rng);
                // Noise up the camera orientation
                cam_pose.segment<3>(3) += random_vector3d(noise, rng);
            }

            auto pose = std::make_shared<PoseVariable>(var_id++, cam_pose);
            if (constant_pose || i == 0) {
                pose->set_constant(true);
            } else if (prior_factors) {
                // Create a pair of prior factors (position / orientation)
                Eigen::Vector3d pos = cam_pose.segment<3>(0);
                Eigen::Vector3d rot = cam_pose.segment<3>(3);
                Eigen::Matrix3d dcm = ExpMapSO3(rot);
                auto position_prior =
                    std::make_shared<PosePositionPriorFactor>(factor_id++, pose.get(), pos, noise_sigma);
                auto rotation_prior =
                    std::make_shared<PoseOrientationPriorFactor>(factor_id++, pose.get(), dcm, noise_sigma);
                graph.add_factor(position_prior);
                graph.add_factor(rotation_prior);
            }
            graph.add_variable(pose);

            poses.push_back(pose);
        }

        std::vector<std::shared_ptr<InverseRangeVariable>> landmarks;

        for (auto lm_pos : landmark_positions) {
            if (random_noise) {
                lm_pos += random_vector3d(noise, rng);
            }
            // initialize all of the range/bearing variables from the first camera pose
            Eigen::Vector3d cam1_pos(camera_poses[0].x(), camera_poses[0].y(), camera_poses[0].z());
            Eigen::Vector3d delta_pos = lm_pos - cam1_pos;
            Eigen::Vector3d bearing_W = (lm_pos - cam1_pos).normalized();

            double range;
            if (initial_range_est > 0.0) {
                range = initial_range_est;
            } else {
                range = delta_pos.norm() + 1.0;
            }
            auto landmark_bearing = std::make_shared<InverseRangeVariable>(var_id++, cam1_pos, bearing_W, range);
            landmarks.push_back(landmark_bearing);
            graph.add_variable(landmark_bearing);
        }

        for (size_t pose_num = 0; pose_num < poses.size(); pose_num++) {
            // Generate bearing measurements for each landmark
            auto pose = poses[pose_num];
            auto dcm_CW = pose->dcm_CW();
            for (size_t lm_num = 0; lm_num < landmarks.size(); lm_num++) {
                auto lm_pos = landmark_positions[lm_num];
                auto lm = landmarks[lm_num];
                Eigen::Vector3d pos_C = dcm_CW * (lm_pos - pose->pos_W());
                Eigen::Vector3d bearing = pos_C.normalized();

                if (random_noise) {
                    bearing += random_vector3d(noise, rng);
                    bearing = bearing.normalized();
                }

                auto factor = std::make_shared<InverseRangeBearingFactor>(factor_id++, pose.get(), lm.get(), bearing,
                                                                          noise_sigma);
                graph.add_factor(factor);
            }
        }
        return graph;
    }

    inline FactorGraph CreateGraphWithPoseBetweenFactors(std::vector<Eigen::Matrix<double, 6, 1>> camera_poses,
                                                         Eigen::Vector3d relative_orientation, bool random_noise,
                                                         double noise_sigma = 0.04)
    {
        FactorGraph graph;
        int var_id = 0;
        int factor_id = 0;

        std::mt19937 rng(42);                                     // fixed seed for reproducibility
        std::normal_distribution<double> noise(0.0, noise_sigma); // small perturbation

        Eigen::Matrix3d dcm_IC = ExpMapSO3(relative_orientation);

        auto extrinsic_rotation_IC = std::make_shared<RotationVariable>(var_id++, Eigen::Matrix3d::Identity());
        graph.add_variable(extrinsic_rotation_IC);

        std::vector<std::shared_ptr<PoseVariable>> poses;
        for (size_t i = 0; i < camera_poses.size(); i++) {
            auto& cam_pose = camera_poses[i];

            Eigen::Matrix3d dcm_CW = ExpMapSO3(cam_pose.segment<3>(3));

            Eigen::Matrix3d dcm_IW = dcm_IC * dcm_CW;

            auto imu_pose = cam_pose;
            imu_pose.segment<3>(3) = LogMapSO3(dcm_IW);

            if (random_noise) {
                // Noise up the imu orientation
                imu_pose.segment<3>(3) += random_vector3d(noise, rng);
            }

            auto cam_pose_var = std::make_shared<PoseVariable>(var_id++, cam_pose);
            cam_pose_var->set_constant(true);
            graph.add_variable(cam_pose_var);

            auto imu_pose_var = std::make_shared<PoseVariable>(var_id++, imu_pose);
            imu_pose_var->set_constant(true);
            graph.add_variable(imu_pose_var);
            // Create pose between factor
            auto factor = std::make_shared<PoseOrientationBetweenFactor>(
                factor_id++, cam_pose_var.get(), imu_pose_var.get(), extrinsic_rotation_IC.get(), noise_sigma);
            graph.add_factor(factor);
        }
        return graph;
    }

    // This is not "the" monotonic clock, but it is "a" monotonic clock
    inline double GetMonotonicSeconds()
    {
        using clock = std::chrono::steady_clock;
        static const auto t_start = clock::now();
        auto t_now = clock::now();
        std::chrono::duration<double> elapsed = t_now - t_start;
        return elapsed.count();
    }

    inline void compare_dense_and_sparse_jacobians(factorama::FactorGraph& graph, double tol = 1e-8)
    {
        std::cout << "\n===== [Jacobian Consistency Check] =====\n";

        // Compute both Jacobians
        Eigen::MatrixXd J_dense = graph.compute_full_jacobian_matrix();
        Eigen::SparseMatrix<double> J_sparse = graph.compute_sparse_jacobian_matrix();

        // Check dimension agreement first
        if (J_dense.rows() != J_sparse.rows() || J_dense.cols() != J_sparse.cols()) {
            std::cerr << "ERROR: Jacobian size mismatch! Dense is " << J_dense.rows() << "x" << J_dense.cols()
                      << ", Sparse is " << J_sparse.rows() << "x" << J_sparse.cols() << "\n";
            return;
        }

        int num_rows = J_dense.rows();
        int num_cols = J_dense.cols();

        // Convert sparse to dense for comparison
        Eigen::MatrixXd J_sparse_dense = Eigen::MatrixXd(J_sparse);

        int num_mismatches = 0;

        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                double val_dense = J_dense(r, c);
                double val_sparse = J_sparse_dense(r, c);
                double diff = std::abs(val_dense - val_sparse);

                if (diff > tol) {
                    ++num_mismatches;
                    std::cout << "Mismatch at (" << r << ", " << c << "): "
                              << "dense = " << val_dense << ", sparse = " << val_sparse << ", diff = " << diff << "\n";
                }
            }
        }

        if (num_mismatches == 0) {
            std::cout << "✅ Jacobians match within tolerance (" << tol << ")\n";
        } else {
            std::cout << "⚠️ Total mismatches: " << num_mismatches << " (tolerance = " << tol << ")\n";
        }

        std::cout << "==========================================\n";
    }

    // Creates a factor graph using BearingProjectionFactor2D instead of BearingObservationFactor
    inline FactorGraph CreateGraphWithBearingProjection2D(std::vector<Eigen::Matrix<double, 6, 1>> camera_poses,
                                                          std::vector<Eigen::Vector3d> landmark_positions,
                                                          bool random_noise, bool constant_pose, bool prior_factors,
                                                          double noise_sigma = 0.04, double sparsity = 0.0)
    {
        FactorGraph graph;
        int var_id = 0;
        int factor_id = 0;

        std::mt19937 rng(42);                                     // fixed seed for reproducibility
        std::normal_distribution<double> noise(0.0, noise_sigma); // small perturbation

        std::vector<std::shared_ptr<PoseVariable>> poses;
        std::vector<std::shared_ptr<PoseVariable>> truth_poses;
        for (size_t i = 0; i < camera_poses.size(); i++) {
            auto original_cam_pose = camera_poses[i];
            auto cam_pose = camera_poses[i];

            if (random_noise && i != 0) {
                // Noise up the camera position (except the first one)
                cam_pose.segment<3>(0) += random_vector3d(noise, rng);
                // Noise up the camera orientation
                cam_pose.segment<3>(3) += random_vector3d(noise, rng);
            }

            auto pose = std::make_shared<PoseVariable>(var_id++, cam_pose);

            if (constant_pose || i == 0) {
                pose->set_constant(true);
            } else if (prior_factors) {
                // Create a pair of prior factors (position / orientation)
                Eigen::Vector3d pos_prior = original_cam_pose.segment<3>(0);
                Eigen::Vector3d rot_prior = original_cam_pose.segment<3>(3);
                Eigen::Matrix3d dcm_prior = ExpMapSO3(rot_prior);

                auto position_prior =
                    std::make_shared<PosePositionPriorFactor>(factor_id++, pose.get(), pos_prior, noise_sigma);
                auto rotation_prior =
                    std::make_shared<PoseOrientationPriorFactor>(factor_id++, pose.get(), dcm_prior, noise_sigma);
                graph.add_factor(position_prior);
                graph.add_factor(rotation_prior);
            }
            graph.add_variable(pose);

            poses.push_back(pose);

            // add truth pose
            auto truth_pose = std::make_shared<PoseVariable>(var_id++, original_cam_pose);
            truth_poses.push_back(truth_pose);
        }
        std::vector<std::shared_ptr<LandmarkVariable>> landmarks;

        for (auto lm_pos : landmark_positions) {
            auto original_lm_pos = lm_pos;
            if (random_noise) {
                lm_pos += random_vector3d(noise, rng);
            }
            auto landmark = std::make_shared<LandmarkVariable>(var_id++, lm_pos);
            landmarks.push_back(landmark);
            graph.add_variable(landmark);
            if (prior_factors) {
                // Create landmark prior factor
                auto landmark_prior =
                    std::make_shared<GenericPriorFactor>(factor_id++, landmark.get(), original_lm_pos, noise_sigma);
                graph.add_factor(landmark_prior);
            }
        }

        for (size_t pose_num = 0; pose_num < poses.size(); pose_num++) {
            // Generate bearing measurements for each landmark
            auto pose = poses[pose_num];
            auto truth_pose = truth_poses[pose_num];
            auto dcm_CW = truth_pose->dcm_CW();
            for (size_t lm_num = 0; lm_num < landmarks.size(); lm_num++) {
                // Randomly choose whether to populate the bearing factor based on "filled_in"
                double tmp = rng();
                tmp /= static_cast<double>(rng.max());
                if (tmp < sparsity) {
                    continue; // Skip this landmark for this camera
                }

                auto lm_pos = landmark_positions[lm_num];
                auto lm = landmarks[lm_num];

                Eigen::Vector3d pos_C = dcm_CW * (lm_pos - truth_pose->pos_W());
                Eigen::Vector3d bearing = pos_C.normalized();

                double angle_noise_mult = 0.1;
                double angle_noise_sigma =
                    noise_sigma * angle_noise_mult; // assume angles sensor is more accurate than position.
                if (random_noise) {
                    bearing += random_vector3d(noise, rng) * angle_noise_mult; // Bearing noise
                    bearing = bearing.normalized();
                }

                // Use BearingProjectionFactor2D instead of BearingObservationFactor
                auto factor = std::make_shared<BearingProjectionFactor2D>(factor_id++, pose.get(), lm.get(), bearing,
                                                                          angle_noise_sigma);
                graph.add_factor(factor);
            }
        }

        return graph;
    }

} // namespace factorama