// testPayloadController.cpp
// Test payload controller — exercises camera states only (no platform/trajectory).
// Platform steps replaced with 3s timers. Auto-starts on launch.

// g++ -std=c++17 payload_comp.cpp -o payload_comp $(pkg-config --cflags --libs lcm)

#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <filesystem>


#include <lcm/lcm-cpp.hpp>
#include "exlcm/payload_cont_to_cam_msg_t.hpp"
#include "exlcm/cam_msg_t.hpp"

// States for the core controller state machine
typedef enum State {
    IDLE,           // Waits for command to start an experiment
    SETUP,          // Placeholder setup step
    CALIBRATE_CAM,  // Focus and calibrate camera
    DEPLOY,         // Deploy platform (simulated) + start camera
    RUNNING,        // Run experiment (simulated)
    SAVE_RESULTS,   // Save results + wait for camera
    TERMINATE_RUN,  // Retract platform (simulated) + stop camera
    ERROR,          // Publish error to camera + return to IDLE
} state_t;


// Auto generated LCM channels and LcmHandler from claude for testing it all in one script
// LCM channel names
static const char* CH_CONT_TO_CAM = "PAYLOAD_CAM";   // controller --> camera
static const char* CH_CAM_TO_CONT = "CAM_PAYLOAD";   // camera --> controller

// LCM message handler
class LcmHandler
{
public:
    bool     cam_status_received = false;
    bool     cam_status          = false;

    void handleCamMsg(const lcm::ReceiveBuffer*, const std::string&,
                      const exlcm::cam_msg_t* msg)
    {
        cam_status_received = true;
        cam_status          = msg->cam_status;
        std::cout << "[INFO] Received cam_msg_t: cam_status=" << msg->cam_status << std::endl;
    }

    // Clear flag before waiting for a new message
    void reset() { cam_status_received = false; cam_status = false; }
};

// Dummy TestPayloadController
class TestPayloadController
{
public:
    TestPayloadController() : lcm_(), handler_()
    {
        if (!lcm_.good())
        {
            std::cout << "[ERROR] LCM object not good." << std::endl;
        }
        lcm_.subscribe(CH_CAM_TO_CONT, &LcmHandler::handleCamMsg, &handler_);
    }

    void run()
    {
        state_t state = IDLE;

        while (true)
        {
            switch (state)
            {
            // ----------------------------------------------------------------
            case IDLE:
                // Note: removed checkRunCommand for testing in one script. Just autostart instead
                std::cout << "[INFO] State: IDLE - auto-starting in 1s..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                state = SETUP;
                std::cout << "[INFO] State set to SETUP." << std::endl;
                break;

            // ----------------------------------------------------------------
            case SETUP:
                std::cout << "[INFO] State: SETUP." << std::endl;
                state = CALIBRATE_CAM;
                std::cout << "[INFO] State set to CALIBRATE_CAM." << std::endl;
                break;

            // ----------------------------------------------------------------
            case CALIBRATE_CAM:
                std::cout << "[INFO] State: CALIBRATE_CAM" << std::endl;

                // Publish CALIBRATE_CAM state to camera with debug_mode = true
                // Note: Change this true to a debug mode flag
                publishCameraCommand(CALIBRATE_CAM, /*debug_mode=*/true);

                // Wait for camera to report complete
                // Note: different to handleTimeout, checkRunCommand setup
                if (!waitForCamStatus())
                {
                    state = ERROR;
                    std::cout << "[INFO] State set to ERROR." << std::endl;
                    break;
                }

                state = DEPLOY;
                std::cout << "[INFO] State set to DEPLOY." << std::endl;
                break;

            // ----------------------------------------------------------------
            case DEPLOY:
                std::cout << "[INFO] State: DEPLOY" << std::endl;
                std::cout << "[INFO] Simulating 3s platform deploy..." << std::endl;                
                std::this_thread::sleep_for(std::chrono::seconds(3));

                // Note: this would go inside platform_deployed block
                // Publish DEPLOY state to camera
                publishCameraCommand(DEPLOY, /*debug_mode=*/false);

                // Wait for camera to report complete
                // Note: different to handleTimeout, checkRunCommand setup
                if (!waitForCamStatus())
                {
                    state = TERMINATE_RUN;
                    std::cout << "[INFO] State set to TERMINATE_RUN." << std::endl;
                    break;
                }

                state = RUNNING;
                std::cout << "[INFO] State set to RUNNING." << std::endl;
                break;

            // ----------------------------------------------------------------
            case RUNNING:
                std::cout << "[INFO] State: RUNNING" << std::endl;

                // Publish RUNNING state to camera
                publishCameraCommand(RUNNING, /*debug_mode=*/false);

                std::cout << "[INFO] Simulating 30s experiment..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(30));

                // Wait for camera to report complete
                if (!waitForCamStatus())
                {
                    state = TERMINATE_RUN;
                    std::cout << "[INFO] State set to TERMINATE_RUN." << std::endl;
                    break;
                }

                state = SAVE_RESULTS;
                std::cout << "[INFO] State set to SAVE_RESULTS." << std::endl;
                break;

            // ----------------------------------------------------------------
            case SAVE_RESULTS:
                std::cout << "[INFO] State: SAVE_RESULTS" << std::endl;

                // Add creating results binary file 
                {
                    std::string results_dir = "outputs/experiment_results";
                    std::filesystem::create_directories(results_dir);

                    std::ofstream results_file(results_dir + "/experiment_results.bin", std::ios::binary);
                    if (!results_file.is_open())
                    {
                        std::cout << "[ERROR] Failed to open results file." << std::endl;
                        state = ERROR;
                        break;
                    }
                }

                // Publish SAVE_RESULTS state to camera
                publishCameraCommand(SAVE_RESULTS, /*debug_mode=*/false);

                // Wait for camera to report complete
                if (!waitForCamStatus())
                {
                    state = ERROR;
                    std::cout << "[INFO] State set to ERROR." << std::endl;
                    break;
                }

                state = IDLE;
                std::cout << "[INFO] State set to IDLE." << std::endl;
                break;

            // ----------------------------------------------------------------
            case TERMINATE_RUN:
                std::cout << "[INFO] State: TERMINATE_RUN" << std::endl;
                std::cout << "[INFO] Simulating 3s platform retract..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(3));

                // Publish TERMINATE_RUN state to camera
                publishCameraCommand(TERMINATE_RUN, /*debug_mode=*/false);

                // Wait for camera to report complete
                if (!waitForCamStatus())
                {
                    state = ERROR;
                    std::cout << "[INFO] State set to ERROR." << std::endl;
                    break;
                }

                state = IDLE;
                std::cout << "[INFO] State set to IDLE." << std::endl;
                break;

            // ----------------------------------------------------------------
            case ERROR:
                std::cout << "[ERROR] State: ERROR." << std::endl;
                std::cout << "[ERROR] Publishing to camera and returning to IDLE." << std::endl;

                // Publish ERROR state to camera
                publishCameraCommand(ERROR, /*debug_mode=*/false);

                state = IDLE;
                std::cout << "[INFO] State set to IDLE." << std::endl;
                break;

            // ----------------------------------------------------------------
            default:
                std::cout << "[ERROR] Invalid state." << std::endl;
                state = IDLE;
                break;
            }
        }
    }

// Note: here for a one script test
private:
    lcm::LCM   lcm_;
    LcmHandler handler_;

    // Timeout when waiting for camera response (ms) (10s)
    static const int CAM_WAIT_TIMEOUT_MS = 10000;

    // Publish current controller state + debug_mode to camera
    void publishCameraCommand(state_t state, bool debug_mode)
    {
        exlcm::payload_cont_to_cam_msg_t msg;
        msg.cont_state = static_cast<int8_t>(state);
        msg.debug_mode = debug_mode;
        lcm_.publish(CH_CONT_TO_CAM, &msg);
        std::cout << "[INFO] Published to " << CH_CONT_TO_CAM
                  << ": cont_state=" << static_cast<int>(state)
                  << " debug_mode=" << debug_mode << std::endl;
    }

    // Block until cam_msg_t received or timeout
    // Returns true if cam_status == true, false on error or timeout
    bool waitForCamStatus()
    {
        handler_.reset();
        auto start = std::chrono::steady_clock::now();

        while (!handler_.cam_status_received)
        {
            lcm_.handleTimeout(100); // poll every 100ms

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();

            if (elapsed >= CAM_WAIT_TIMEOUT_MS)
            {
                std::cout << "[ERROR] Timed out waiting for camera response." << std::endl;
                return false;
            }
        }

        if (!handler_.cam_status)
        {
            std::cout << "[ERROR] Camera reported failure (cam_status=false)." << std::endl;
            return false;
        }

        std::cout << "[INFO] Camera reported success." << std::endl;
        return true;
    }
};

// Main
int main()
{
    std::cout << "[INFO] Starting test payload controller..." << std::endl;
    TestPayloadController controller;
    controller.run();
    return 0;
}













// // ORIGINAL FROM LINCOLN'S REPO
// // ---------------------------------------------------------------------------------------------------------
// #include "payloadController.hpp"
// #include "commands.hpp"

// #include <iostream>
// #include <fstream>
// #include <sstream>

// #define TRAJECTORY_FILE_STEP 500 // ms, time between successive poses in the trajectory file
// #define TRAJECTORY_STRUCT_STEP 10 // ms, time between successive poses in the trajectory struct
// #define TRAJECTORY_FILE_PATH "../../../data/trajectory_simple.csv"

// // States for the core controller state machine
// typedef enum State {
//     IDLE,           // Waits for command to start an experiment
//     SETUP,          // Reads the trajectory file + calibrates servos
//     DEPLOY,         // Deploys the docking port to the starting position
//     RUNNING,        // Runs the experiment
//     SAVE_RESULTS,   // Saves experiment data and tells the camera node to do the same
//     TERMINATE_RUN,  // Moves the platform back to the home position 
//     ERROR,          // Publishes an erroneous run result
// } state_t;


// PayloadController::PayloadController()
//     : lcm(), lcm_handler(), error(), platform(), trajectory_step(0), experiment_start_time()
// {
//     trajectory_path = TRAJECTORY_FILE_PATH;

//     if (!lcm.good())
//     {
//         std::cout << "[ERROR] Payload LCM object not good." << std::endl;
//     }

//     // Subscribe lcm handler to messages
//     lcm.subscribe("RUN_COMMAND", &LcmHandler::handleRunCommand, &lcm_handler);
//     lcm.subscribe("SAVE_COMPLETE", &LcmHandler::handleSaveComplete, &lcm_handler);
// };

// PayloadController::~PayloadController(){};

// void PayloadController::run()
// {
//     state_t state = IDLE;
//     int command_id;
//     bool platform_deployed, trajectory_complete;
    
//     while (true)
//     {
//         switch (state)
//         {
//         case IDLE:
//             // Check if a run command has been published
//             lcm.handleTimeout(0);
//             if (lcm_handler.checkRunCommand(command_id))
//             {
//                 // Only move to setup when start command received
//                 if (command_id == Commands::RunId::RUN_CONTROLLER)
//                 {
//                     state = SETUP;
//                     std::cout << "[INFO] Payload controller state set to SETUP." << std::endl;
//                     break;
//                 }
//             }

//             break;
//         case SETUP:
//             // Read the trajectory file, interpolate, and compute servo angles
//             if (buildTrajectory() == false)
//             {
//                 state = ERROR;
//                 std::cout << "[INFO] Payload controller state set to ERROR." << std::endl;
//                 break;
//             }

//             // Calibrate the servos
//             // TODO

//             // Automatically transition to deploy when the setup steps succeed
//             state = DEPLOY;
//             std::cout << "[INFO] Payload controller state set to DEPLOY." << std::endl;
//             break; 
//         case DEPLOY: 
//             // Make an incremental step to move the platform to the starting position
//             if (deployPlatformStep(platform_deployed) == false)
//             {
//                 error.msg = "Could not deploy platform.";
//                 state = ERROR;
//                 std::cout << "[INFO] Payload controller state set to ERROR." << std::endl;
//                 break;
//             }
            
//             // Check if a message has been published to stop the experiment early
//             lcm.handleTimeout(0);
//             if (lcm_handler.checkRunCommand(command_id))
//             {
//                 if (command_id == Commands::RunId::STOP_CONTROLLER)
//                 {
//                     state = TERMINATE_RUN;
//                     std::cout << "[INFO] Payload controller state set to TERMINATE_RUN." << std::endl;
//                     break;
//                 }
//             }

//             // Check if the platform is fully deployed before moving to RUNNING
//             if (platform_deployed == true)
//             {
//                 // Start camera nodes
//                 publishCameraCommand(Commands::CameraCommandId::START_CAMERA);

//                 state = RUNNING;
//                 std::cout << "[INFO] Payload controller state set to RUNNING." << std::endl;
//                 experiment_start_time = std::chrono::steady_clock::now(); // Start timing experiment
//                 trajectory_step = 0; // Track trajectory from the start 
//                 break;
//             }

//             break;
//         case RUNNING: 
//             // Make an incremental step to move the platform along the trajectory
//             if (trackTrajectoryStep(trajectory_complete) == false)
//             {
//                 error.msg = "Failure while tracking trajectory.";
//                 state = ERROR;
//                 std::cout << "[INFO] Payload controller state set to ERROR." << std::endl;
//                 break;
//             }

//             // Check if a message has been published to stop the experiment early
//             lcm.handleTimeout(0);
//             if (lcm_handler.checkRunCommand(command_id))
//             {
//                 if (command_id == Commands::RunId::STOP_CONTROLLER)
//                 {
//                     state = TERMINATE_RUN;
//                     std::cout << "[INFO] Payload controller state set to TERMINATE_RUN." << std::endl;
//                     break;
//                 }
//             }

//             // Check if the trajectory is complete before moving to SAVE_RESULTS
//             if (trajectory_complete == true)
//             {
//                 state = SAVE_RESULTS;
//                 std::cout << "[INFO] Payload controller state set to SAVE_RESULTS." << std::endl;
//                 break;
//             }

//             break;
//         case SAVE_RESULTS:
//             // Create a results file and save the servo angles across the trajectory
//             // TODO

//             // Prompt the event camera node to save its results and wait for confirmation
//             publishCameraCommand(Commands::CameraCommandId::STOP_AND_SAVE);

//             if (waitForSaveComplete() == false)
//             {
//                 state = ERROR;
//                 std::cout << "[INFO] Payload controller state set to ERROR." << std::endl;
//                 break;
//             }

//             // Let the OBC bridge know the experiment is complete and results file has been saved
//             publishRunResult(Commands::RunResult::RUN_SUCCESS);

//             state = IDLE;
//             std::cout << "[INFO] Payload controller state set to IDLE." << std::endl;
//             break;
//         case TERMINATE_RUN:
//             if (retractPlatform() == false)
//             {
//                 error.msg = "Failed to retract the platform automatically.";
//                 state = ERROR;
//                 std::cout << "[INFO] Payload controller state set to ERROR." << std::endl;
//                 break;
//             }

//             // Automatically move back to IDLE
//             state = IDLE;
//             std::cout << "[INFO] Payload controller state set to IDLE." << std::endl;
//             break;
//         case ERROR: 
//             // TODO: determine if the error message should be used elsewhere
//             std::cout << "[ERROR] " << error.msg << std::endl;
            
//             // Let the OBC bridge know the experiment failed
//             publishRunResult(Commands::RunResult::RUN_FAIL);

//             // Automatically move back to IDLE
//             state = IDLE;
//             std::cout << "[INFO] Payload controller state set to IDLE." << std::endl;
//             break;
//         default: 
//             std::cout << "[ERROR] Payload controller entered invalid state." << std::endl;
//         }
//     }
// }

// // --- LCM publisher methods ---------------------------------------------------

// void PayloadController::publishCameraCommand(int8_t command_id)
// {
//     payload_messages::camera_command_t msg;
//     msg.command_id = command_id;
//     std::cout << "[INFO] Publishing to CAMERA_COMMAND: command_id=" << (int)command_id << std::endl;
//     lcm.publish("CAMERA_COMMAND", &msg);
// }

// void PayloadController::publishRunResult(int8_t return_id)
// {
//     payload_messages::run_result_t msg;
//     msg.return_id = return_id;
//     std::cout << "[INFO] Publishing to RUN_RESULT: return_id=" << (int)return_id << std::endl;
//     lcm.publish("RUN_RESULT", &msg);
// }




// // // --- Trajectory tracking -----------------------------------------------------

// // bool PayloadController::deployPlatformStep(bool &platform_deployed)
// // {
// //     // TODO: current set to move past successfully
// //     platform_deployed = true;

// //     return true; 
// // }

// // bool PayloadController::trackTrajectoryStep(bool &trajectory_complete)
// // {
// //     trajectory_complete = false;

// //     // Check if there are no more poses remaining to track
// //     if (trajectory_step == trajectory.times.size())
// //     {
// //         trajectory_complete = true;
// //         return true;
// //     }
    
// //     // Get the current experiment time in ms
// //     std::chrono::duration<double> time_temp = std::chrono::steady_clock::now() - experiment_start_time;
// //     double experiment_time = std::chrono::duration<double, std::milli>(time_temp).count();

// //     // Move the platform to the next pose until we catch up to the current time
// //     while (experiment_time >= trajectory.times[trajectory_step])
// //     {
// //         if (platform.moveTo(trajectory.poses[trajectory_step]) == false)
// //         {
// //             error.msg = "Could not move platform to target pose.";
// //             return false;
// //         }
// //         trajectory_step++;
// //     }    

// //     return true; 
// // }

// // bool PayloadController::retractPlatform()
// // {
// //     // TODO
// //     return false;
// // }

// // bool PayloadController::waitForSaveComplete()
// // {
// //     bool result = false;
// //     int return_id;

// //     while (lcm.getFileno() >= 0)
// //     {
// //         lcm.handle();

// //         if (lcm_handler.checkSaveComplete(return_id))
// //         {
// //             if (return_id == Commands::SaveResult::SAVE_SUCCESS)
// //                 result = true;
// //             else if (return_id == Commands::SaveResult::SAVE_FAIL)
// //                 error.msg = "Camera node failed to save results.";
// //             else
// //                 error.msg = "Unknown return_id published to SAVE_COMPLETE.";

// //             break;
// //         }
// //     }

// //     return result;
// // }

// // // --- File i/o ----------------------------------------------------------------

// // bool PayloadController::readRawPoses(std::vector<PlatformPose>& raw_poses)
// // {
// //     bool result = true;
// //     std::ifstream file(trajectory_path);

// //     // Check the file could be opened
// //     if (!file.is_open())
// //     {
// //         std::cout << "Trajectory file not found." << std::endl;
// //         result = false;
// //     }
// //     else
// //     {
// //         // Read each line of the file
// //         std::string line;
// //         while (std::getline(file, line))
// //         {
// //             std::stringstream ss(line);
// //             std::string p_x, p_y, p_z, roll, pitch, yaw;

// //             // Parse the six comma-separated values on each line
// //             if (std::getline(ss, p_x, ',') &&
// //                 std::getline(ss, p_y, ',') &&
// //                 std::getline(ss, p_z, ',') &&
// //                 std::getline(ss, roll, ',') &&
// //                 std::getline(ss, pitch, ',') &&
// //                 std::getline(ss, yaw))
// //             {
// //                 // Convert euler angles in degrees to a quaternion
// //                 Eigen::AngleAxisf roll_angle(std::stof(roll) * M_PI / 180,   Eigen::Vector3f::UnitX());
// //                 Eigen::AngleAxisf pitch_angle(std::stof(pitch) * M_PI / 180, Eigen::Vector3f::UnitY());
// //                 Eigen::AngleAxisf yaw_angle(std::stof(yaw) * M_PI / 180,     Eigen::Vector3f::UnitZ());

// //                 Eigen::Quaternionf q = yaw_angle * pitch_angle * roll_angle;

// //                 PlatformPose pose {
// //                     Vector3f(std::stof(p_x), std::stof(p_y), std::stof(p_z)), q
// //                 };

// //                 raw_poses.push_back(pose);
// //             }
// //             else
// //             {
// //                 std::cout << "Error reading trajectory file." << std::endl;
// //                 result = false;
// //                 break;
// //             }
// //         }

// //         file.close();
// //     }

// //     return result;
// // }

// // bool PayloadController::writeAnglesToFile(std::string file_path)
// // {
// //     bool result = true;
// //     std::ofstream file(file_path);

// //     // Check the file could be opened/created
// //     if (!file.is_open())
// //     {
// //         std::cout << "Error: could not open file for writing: " << file_path << std::endl;
// //         result = false;
// //     }
// //     else
// //     {
// //         // Write each row of servo angles as a comma-separated line
// //         for (const auto& angles : trajectory.angles)
// //         {
// //             for (size_t i = 0; i < NUM_SERVOS - 1; i++)
// //             {
// //                 file << angles[i] << ",";
// //             }

// //             file << angles[NUM_SERVOS - 1] << "\n";
// //         }

// //         file.close();
// //     }

// //     return result;
// // }

// // // --- Trajectory building -----------------------------------------------------

// // bool PayloadController::interpolateTrajectory(const std::vector<PlatformPose>& raw_poses, trajectory_t& out)
// // {
// //     // Require at least two poses to interpolate between
// //     if (raw_poses.size() < 2)
// //     {
// //         return false;
// //     }

// //     const int n_steps = TRAJECTORY_FILE_STEP / TRAJECTORY_STRUCT_STEP;

// //     // Interpolate between each consecutive pair of raw poses
// //     for (size_t i = 0; i < raw_poses.size() - 1; i++)
// //     {
// //         // Generate n_steps interpolated poses between raw_poses[i] and raw_poses[i+1]
// //         for (int j = 0; j < n_steps; j++)
// //         {
// //             float t = (float)j / n_steps;

// //             // Linearly interpolate position
// //             Vector3f pos = raw_poses[i].position + t * (raw_poses[i + 1].position - raw_poses[i].position);

// //             // Spherically interpolate orientation
// //             Eigen::Quaternionf orientation = raw_poses[i].orientation.slerp(t, raw_poses[i + 1].orientation);

// //             out.poses.push_back({pos, orientation});
// //             out.times.push_back((float)(i * TRAJECTORY_FILE_STEP + j * TRAJECTORY_STRUCT_STEP));
// //         }
// //     }

// //     // Append the final raw pose to close the trajectory
// //     out.poses.push_back(raw_poses.back());
// //     out.times.push_back((float)((raw_poses.size() - 1) * TRAJECTORY_FILE_STEP));

// //     return true;
// // }

// // bool PayloadController::computeTrajectoryAngles(trajectory_t& traj)
// // {
// //     bool result = true;
// //     std::array<float, NUM_SERVOS> angles;

// //     // Pre-allocate memory for the angles
// //     traj.angles.resize(traj.poses.size());

// //     // Compute the required servo angles for each pose in the trajectory
// //     for (size_t i = 0; i < traj.poses.size(); i++)
// //     {
// //         if (!platform.getAnglesForMove(traj.poses[i], &angles))
// //         {
// //             result = false;
// //             break;
// //         }

// //         traj.angles[i] = angles;
// //     }

// //     return result;
// // }

// // bool PayloadController::buildTrajectory()
// // {
// //     trajectory_t temp;
// //     std::vector<PlatformPose> raw_poses;

// //     // Read the raw poses from the trajectory file
// //     if (readRawPoses(raw_poses) == false)
// //     {
// //         error.msg = "Could not read trajectory file.";
// //         return false;
// //     }

// //     // Interpolate between raw poses at TRAJECTORY_STRUCT_STEP intervals
// //     if (interpolateTrajectory(raw_poses, temp) == false)
// //     {
// //         error.msg = "Could not interpolate trajectory.";
// //         return false;
// //     }

// //     // Compute servo angles for each interpolated pose
// //     if (computeTrajectoryAngles(temp) == false)
// //     {
// //         error.msg = "Could not convert trajectory to servo angles.";
// //         return false;
// //     }

// //     // Assign only on complete success to avoid partial population
// //     trajectory = temp;
// //     return true;
// // }

// // // --- Trajectory debugging ----------------------------------------------------

// // bool PayloadController::generateTrajectoryAnglesFile(std::string file_path)
// // {
// //     bool result = true;

// //     // Build the full trajectory struct from the trajectory file
// //     if (buildTrajectory() == false)
// //     {
// //         result = false;
// //     }

// //     // Write the computed servo angles to the output file
// //     if (result == true && writeAnglesToFile(file_path) == false)
// //     {
// //         result = false;
// //     }

// //     if (result == false)
// //     {
// //         std::cout << "Error: could not generate trajectory angles file." << std::endl;
// //     }

// //     return result;
// // }

// // void PayloadController::printTrajectory()
// // {
// //     for (size_t i = 0; i < trajectory.poses.size(); i++)
// //     {
// //         const PlatformPose& pose = trajectory.poses[i];

// //         // Print timestamp, position, and orientation
// //         std::cout
// //             << "t=" << trajectory.times[i] << " ms  "
// //             << "pos=["  << pose.position.transpose() << "]  "
// //             << "ori=["  << pose.orientation.w() << " "
// //                         << pose.orientation.x() << "i "
// //                         << pose.orientation.y() << "j "
// //                         << pose.orientation.z() << "k]  "
// //             << "angles=[";

// //         // Print servo angles
// //         for (size_t j = 0; j < NUM_SERVOS; j++)
// //         {
// //             std::cout << trajectory.angles[i][j];
// //             if (j < NUM_SERVOS - 1)
// //             {
// //                 std::cout << " ";
// //             }
// //         }

// //         std::cout << "]" << std::endl;
// //     }
// // };

