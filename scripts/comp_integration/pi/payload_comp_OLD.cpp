// #include "payloadController.hpp"
// #include "commands.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>  // sleep()

// Added in for camera test integration
#include <lcm/lcm-cpp.hpp>
#include "exlcm/payload_comp_msg_t.hpp"
#include "exlcm/cam_msg_t.hpp"
#include <string>
#include <vector>
#include <chrono>

#define TRAJECTORY_FILE_STEP 500
#define TRAJECTORY_STRUCT_STEP 10
#define TRAJECTORY_FILE_PATH "../../../data/trajectory_simple.csv"

// States for the core controller state machine
typedef enum State {
    IDLE,           // Waits for command to start an experiment
    SETUP,          // Reads the trajectory file + calibrates servos
    DEPLOY,         // Deploys the docking port to the starting position
    RUNNING,        // Runs the experiment
    SAVE_RESULTS,   // Saves experiment data and tells the camera node to do the same
    TERMINATE_RUN,  // Moves the platform back to the home position
    ERROR,          // Publishes an erroneous run result
} state_t;

// ----------------------------------------------------------------------------------
// Camera message handler 
class CamHandler
{
public:
    bool received = false;
    exlcm::cam_msg_t last_msg;

    void handleMessage(const lcm::ReceiveBuffer* /*rbuf*/,
                       const std::string& channel,
                       const exlcm::cam_msg_t* msg)
    {
        last_msg = *msg;
        received = true;
        std::cout << "[CAM] Received message on channel \"" << channel << "\"" << std::endl;
        std::cout << "      cam_calib_complete = "
                  << (msg->cam_calib_complete ? "True" : "False") << std::endl;
        std::cout << "      exp_complete       = "
                  << (msg->exp_complete ? "True" : "False") << std::endl;
        std::cout << std::endl;
    }
};

// Wait until a msg from camera, or time out
static bool wait_for_cam_msg(lcm::LCM& lc, exlcm::cam_msg_t& out_msg,
                              int timeout_ms = 3000)
{
    CamHandler handler;
    lcm::Subscription* sub = lc.subscribe("PAYLOAD_CAM",
                                           &CamHandler::handleMessage,
                                           &handler);
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(timeout_ms);
    while (!handler.received && std::chrono::steady_clock::now() < deadline)
        lc.handleTimeout(50);

    lc.unsubscribe(sub);

    if (handler.received) {
        out_msg = handler.last_msg;
        return true;
    }
    std::cout << "[CAM] No message received within " << timeout_ms
              << " ms — passing through." << std::endl;
    return false;
}

// Publish a payload_comp_msg_t to PAYLOAD_CAM
static void publish_payload_comp_msg(lcm::LCM& lc,
                                     bool cam_enabled = false,
                                     bool exp_enabled = false)
{
    exlcm::payload_comp_msg_t msg;
    msg.cam_enabled = cam_enabled;
    msg.exp_enabled = exp_enabled;
    lc.publish("PAYLOAD_CAM", &msg);
    std::cout << "[CAM] Published to PAYLOAD_CAM  cam_enabled=" << cam_enabled
              << "  exp_enabled=" << exp_enabled << std::endl;
}
// ----------------------------------------------------------------------------------
// ── Dummy LcmHandler ─────────────────────────────────────────────────────────
// Stub for RUN_COMMAND and SAVE_COMPLETE subscriptions (not under test)
struct LcmHandler {
    int  run_command_id    = -1;
    bool run_command_ready = false;
    int  save_result_id    = -1;
    bool save_result_ready = false;

    void handleRunCommand(const lcm::ReceiveBuffer* /*rbuf*/,
                          const std::string& /*channel*/,
                          const payload_messages::run_command_t* msg)
    {
        run_command_id    = msg->command_id;
        run_command_ready = true;
    }

    void handleSaveComplete(const lcm::ReceiveBuffer* /*rbuf*/,
                            const std::string& /*channel*/,
                            const payload_messages::save_complete_t* msg)
    {
        save_result_id    = msg->return_id;
        save_result_ready = true;
    }

    bool checkRunCommand(int& id) {
        if (run_command_ready) { id = run_command_id; run_command_ready = false; return true; }
        return false;
    }

    bool checkSaveComplete(int& id) {
        if (save_result_ready) { id = save_result_id; save_result_ready = false; return true; }
        return false;
    }
};


// ── Dummy error struct ────────────────────────────────────────────────────────
struct ErrorState {
    std::string msg;
};

// Dummy Payload Controller
// Holds only the members needed for camera integration testing.
// Platform, trajectory, and servo members are stubbed or omitted.
class PayloadController {
public:
    lcm::LCM   lcm;
    LcmHandler lcm_handler;
    ErrorState error;

    // Trajectory / platform stubs — kept so commented-out code can be
    // uncommented without changing the class definition
    int         trajectory_step = 0;
    std::string trajectory_path;
    std::chrono::time_point<std::chrono::steady_clock> experiment_start_time;

    PayloadController();
    ~PayloadController();
    void run();
};

// -----------------------------------------------------------------------------------


// =============================================================================

PayloadController::PayloadController()
    : lcm(), lcm_handler(), error(), platform(), trajectory_step(0), experiment_start_time()
{
    trajectory_path = TRAJECTORY_FILE_PATH;
    if (!lcm.good())
        std::cout << "[ERROR] Payload LCM object not good." << std::endl;

    lcm.subscribe("RUN_COMMAND",   &LcmHandler::handleRunCommand,   &lcm_handler);
    lcm.subscribe("SAVE_COMPLETE", &LcmHandler::handleSaveComplete, &lcm_handler);
};

PayloadController::~PayloadController(){};

void PayloadController::run()
{
    state_t state = IDLE;
    int command_id;
    bool platform_deployed, trajectory_complete;

    while (true)
    {
        switch (state)
        {
        case IDLE:
            std::cout << "[INFO] State: IDLE" << std::endl;

            sleep(3);
            
            // TODO Uncomment trajectory functionality
            // lcm.handleTimeout(0);
            // if (lcm_handler.checkRunCommand(command_id))
            // {
            //     if (command_id == Commands::RunId::RUN_CONTROLLER)
            //     {
            //         state = SETUP;
            //         std::cout << "[INFO] Payload controller state set to SETUP." << std::endl;
            //         break;
            //     }
            // }

            // Just for camera test integration
            state = SETUP;
            std::cout << "[INFO] Payload controller state set to SETUP." << std::endl;
            break;
        case SETUP:
            std::cout << "[INFO] State: SETUP" << std::endl;

            // TODO Uncomment trajectory functionality
            // // Read the trajectory file, interpolate, and compute servo angles
            // if (buildTrajectory() == false)
            // {
            //     state = ERROR;
            //     std::cout << "[INFO] Payload controller state set to ERROR." << std::endl;
            //     break;
            // }

            // // Calibrate the servos
            // // TODO

            // // Automatically transition to deploy when the setup steps succeed
            // state = DEPLOY;
            // std::cout << "[INFO] Payload controller state set to DEPLOY." << std::endl;


            // Send START_CAMERA and wait for calibration confirmation
            // TODO Fix this communication/messages
            publish_payload_comp_msg(lcm, /*cam_enabled=*/true);
            {
                exlcm::cam_msg_t reply;
                if (wait_for_cam_msg(lcm, reply))
                    std::cout << "[SETUP] Camera calibration confirmed" << std::endl;
                else
                    state = ERROR;
                    std::cout << "[SETUP] No camera reply - Payload controller state set to ERROR." << std::endl;
                    break;
            }

            // Just for camera test integration
            state = DEPLOY;
            std::cout << "[INFO] Payload controller state set to DEPLOY." << std::endl;
            break;
        case DEPLOY:
            std::cout << "[INFO] State: DEPLOY " << std::endl;
            
            sleep(3);
            
            // TODO Uncomment trajectory functionality
            // // Make an incremental step to move the platform to the starting position
            // if (deployPlatformStep(platform_deployed) == false)
            // {
            //     error.msg = "Could not deploy platform.";
            //     state = ERROR;
            //     std::cout << "[INFO] Payload controller state set to ERROR." << std::endl;
            //     break;
            // }
            
            // // Check if a message has been published to stop the experiment early
            // lcm.handleTimeout(0);
            // if (lcm_handler.checkRunCommand(command_id))
            // {
            //     if (command_id == Commands::RunId::STOP_CONTROLLER)
            //     {
            //         state = TERMINATE_RUN;
            //         std::cout << "[INFO] Payload controller state set to TERMINATE_RUN." << std::endl;
            //         break;
            //     }
            // }

            // // Check if the platform is fully deployed before moving to RUNNING
            // if (platform_deployed == true)
            // {
            //     // Start camera nodes
            //     publishCameraCommand(Commands::CameraCommandId::START_CAMERA);

            //     state = RUNNING;
            //     std::cout << "[INFO] Payload controller state set to RUNNING." << std::endl;
            //     experiment_start_time = std::chrono::steady_clock::now(); // Start timing experiment
            //     trajectory_step = 0; // Track trajectory from the start 
            //     break;
            // }
            
            // Just for camera test integration
            state = RUNNING;
            std::cout << "[INFO] Payload controller state set to RUNNING." << std::endl;

            break;
        case RUNNING:
            std::cout << "[INFO] State: RUNNING" << std::endl;

            // Send START_EXP and wait for experiment-complete confirmation
            // TODO check this communication/messages
            publish_payload_comp_msg(lcm, /*cam_enabled=*/false, /*exp_enabled=*/true);
            {
                exlcm::cam_msg_t reply;
                if (wait_for_cam_msg(lcm, reply)) {
                    std::cout << "[RUNNING] Experiment complete confirmed." << std::endl;
                    trajectory_complete = reply.exp_complete;
                } else {
                    std::cout << "[RUNNING] No camera reply — treating experiment as complete." << std::endl;
                    trajectory_complete = true;
                }
            }

            // TODO Uncomment trajectory functionality
            // lcm.handleTimeout(0);
            // if (lcm_handler.checkRunCommand(command_id))
            // {
            //     if (command_id == Commands::RunId::STOP_CONTROLLER)
            //     {
            //         state = TERMINATE_RUN;
            //         std::cout << "[INFO] Payload controller state set to TERMINATE_RUN." << std::endl;
            //         break;
            //     }
            // }

            // if (trajectory_complete == true)
            // {
            //     state = SAVE_RESULTS;
            //     std::cout << "[INFO] Payload controller state set to SAVE_RESULTS." << std::endl;
            //     break;
            // }
            
            // Just for camera test integration
            state = SAVE_RESULTS;
            std::cout << "[INFO] Payload controller state set to SAVE_RESULTS." << std::endl;
            break;
        case SAVE_RESULTS:
            std::cout << "[INFO] State: SAVE_RESULTS" << std::endl;

            sleep(3);
            
            // TODO Uncomment trajectory functionality
            // TODO Camera already automatically saves results after experiment, so no need for this flag
            // // Prompt the event camera node to save its results and wait for confirmation
            // publishCameraCommand(Commands::CameraCommandId::STOP_AND_SAVE);

            // if (waitForSaveComplete() == false)
            // {
            //     state = ERROR;
            //     std::cout << "[INFO] Payload controller state set to ERROR." << std::endl;
            //     break;
            // }

            // // Let the OBC bridge know the experiment is complete and results file has been saved
            // publishRunResult(Commands::RunResult::RUN_SUCCESS);

            // Just for camera test integration
            state = IDLE;
            std::cout << "[INFO] Payload controller state set to IDLE." << std::endl;
            break;
        case TERMINATE_RUN:
            std::cout << "[INFO] State: TERMINATE_RUN" << std::endl;
            sleep(3);

            // TODO Uncomment trajectory functionality
            // if (retractPlatform() == false)
            // {
            //     error.msg = "Failed to retract the platform automatically.";
            //     state = ERROR;
            //     std::cout << "[INFO] Payload controller state set to ERROR." << std::endl;
            //     break;
            // }

            // Automatically move back to IDLE
            state = IDLE;
            std::cout << "[INFO] Payload controller state set to IDLE." << std::endl;
            break;
        case ERROR:
            std::cout << "[INFO] State: ERROR" << std::endl;

            // TODO Uncomment trajectory functionality
            // // TODO: determine if the error message should be used elsewhere
            // std::cout << "[ERROR] " << error.msg << std::endl;
            
            // // Let the OBC bridge know the experiment failed
            // publishRunResult(Commands::RunResult::RUN_FAIL);

            state = IDLE;
            std::cout << "[INFO] Payload controller state set to IDLE." << std::endl;
            break;

        default:
            std::cout << "[ERROR] Payload controller entered invalid state." << std::endl;
        }
    }
}


int main()
{
    PayloadController payload;
    payload.run();

    // payload.generateTrajectoryAnglesFile("../data/angles.csv");

    // StewartPlatformAnalyser platform_analyser; 
    // platform_analyser.generatePointCloud("../data/point_cloud.csv");
}



// TODO Uncomment trajectory functionality
// // --- Trajectory tracking -----------------------------------------------------

// bool PayloadController::deployPlatformStep(bool &platform_deployed)
// {
//     platform_deployed = true;
//     return true;
// }

// bool PayloadController::trackTrajectoryStep(bool &trajectory_complete)
// {
//     trajectory_complete = false;
//     if (trajectory_step == trajectory.times.size())
//     {
//         trajectory_complete = true;
//         return true;
//     }
//     std::chrono::duration<double> time_temp = std::chrono::steady_clock::now() - experiment_start_time;
//     double experiment_time = std::chrono::duration<double, std::milli>(time_temp).count();
//     while (experiment_time >= trajectory.times[trajectory_step])
//     {
//         if (platform.moveTo(trajectory.poses[trajectory_step]) == false)
//         {
//             error.msg = "Could not move platform to target pose.";
//             return false;
//         }
//         trajectory_step++;
//     }
//     return true;
// }

// bool PayloadController::retractPlatform()
// {
//     // TODO
//     return false;
// }

// bool PayloadController::waitForSaveComplete()
// {
//     bool result = false;
//     int return_id;
//     while (lcm.getFileno() >= 0)
//     {
//         lcm.handle();
//         if (lcm_handler.checkSaveComplete(return_id))
//         {
//             if (return_id == Commands::SaveResult::SAVE_SUCCESS)
//                 result = true;
//             else if (return_id == Commands::SaveResult::SAVE_FAIL)
//                 error.msg = "Camera node failed to save results.";
//             else
//                 error.msg = "Unknown return_id published to SAVE_COMPLETE.";
//             break;
//         }
//     }
//     return result;
// }

// // --- File i/o ----------------------------------------------------------------

// bool PayloadController::readRawPoses(std::vector<PlatformPose>& raw_poses)
// {
//     bool result = true;
//     std::ifstream file(trajectory_path);
//     if (!file.is_open())
//     {
//         std::cout << "Trajectory file not found." << std::endl;
//         result = false;
//     }
//     else
//     {
//         std::string line;
//         while (std::getline(file, line))
//         {
//             std::stringstream ss(line);
//             std::string p_x, p_y, p_z, roll, pitch, yaw;
//             if (std::getline(ss, p_x, ',') &&
//                 std::getline(ss, p_y, ',') &&
//                 std::getline(ss, p_z, ',') &&
//                 std::getline(ss, roll, ',') &&
//                 std::getline(ss, pitch, ',') &&
//                 std::getline(ss, yaw))
//             {
//                 Eigen::AngleAxisf roll_angle(std::stof(roll) * M_PI / 180,   Eigen::Vector3f::UnitX());
//                 Eigen::AngleAxisf pitch_angle(std::stof(pitch) * M_PI / 180, Eigen::Vector3f::UnitY());
//                 Eigen::AngleAxisf yaw_angle(std::stof(yaw) * M_PI / 180,     Eigen::Vector3f::UnitZ());
//                 Eigen::Quaternionf q = yaw_angle * pitch_angle * roll_angle;
//                 PlatformPose pose {
//                     Vector3f(std::stof(p_x), std::stof(p_y), std::stof(p_z)), q
//                 };
//                 raw_poses.push_back(pose);
//             }
//             else
//             {
//                 std::cout << "Error reading trajectory file." << std::endl;
//                 result = false;
//                 break;
//             }
//         }
//         file.close();
//     }
//     return result;
// }

// bool PayloadController::writeAnglesToFile(std::string file_path)
// {
//     bool result = true;
//     std::ofstream file(file_path);
//     if (!file.is_open())
//     {
//         std::cout << "Error: could not open file for writing: " << file_path << std::endl;
//         result = false;
//     }
//     else
//     {
//         for (const auto& angles : trajectory.angles)
//         {
//             for (size_t i = 0; i < NUM_SERVOS - 1; i++)
//                 file << angles[i] << ",";
//             file << angles[NUM_SERVOS - 1] << "\n";
//         }
//         file.close();
//     }
//     return result;
// }

// // --- Trajectory building -----------------------------------------------------

// bool PayloadController::interpolateTrajectory(const std::vector<PlatformPose>& raw_poses, trajectory_t& out)
// {
//     if (raw_poses.size() < 2)
//         return false;

//     const int n_steps = TRAJECTORY_FILE_STEP / TRAJECTORY_STRUCT_STEP;
//     for (size_t i = 0; i < raw_poses.size() - 1; i++)
//     {
//         for (int j = 0; j < n_steps; j++)
//         {
//             float t = (float)j / n_steps;
//             Vector3f pos = raw_poses[i].position + t * (raw_poses[i + 1].position - raw_poses[i].position);
//             Eigen::Quaternionf orientation = raw_poses[i].orientation.slerp(t, raw_poses[i + 1].orientation);
//             out.poses.push_back({pos, orientation});
//             out.times.push_back((float)(i * TRAJECTORY_FILE_STEP + j * TRAJECTORY_STRUCT_STEP));
//         }
//     }
//     out.poses.push_back(raw_poses.back());
//     out.times.push_back((float)((raw_poses.size() - 1) * TRAJECTORY_FILE_STEP));
//     return true;
// }

// bool PayloadController::computeTrajectoryAngles(trajectory_t& traj)
// {
//     bool result = true;
//     std::array<float, NUM_SERVOS> angles;
//     traj.angles.resize(traj.poses.size());
//     for (size_t i = 0; i < traj.poses.size(); i++)
//     {
//         if (!platform.getAnglesForMove(traj.poses[i], &angles))
//         {
//             result = false;
//             break;
//         }
//         traj.angles[i] = angles;
//     }
//     return result;
// }

// bool PayloadController::buildTrajectory()
// {
//     trajectory_t temp;
//     std::vector<PlatformPose> raw_poses;
//     if (readRawPoses(raw_poses) == false)
//     {
//         error.msg = "Could not read trajectory file.";
//         return false;
//     }
//     if (interpolateTrajectory(raw_poses, temp) == false)
//     {
//         error.msg = "Could not interpolate trajectory.";
//         return false;
//     }
//     if (computeTrajectoryAngles(temp) == false)
//     {
//         error.msg = "Could not convert trajectory to servo angles.";
//         return false;
//     }
//     trajectory = temp;
//     return true;
// }

// // --- Trajectory debugging ----------------------------------------------------

// bool PayloadController::generateTrajectoryAnglesFile(std::string file_path)
// {
//     bool result = true;
//     if (buildTrajectory() == false)
//         result = false;
//     if (result == true && writeAnglesToFile(file_path) == false)
//         result = false;
//     if (result == false)
//         std::cout << "Error: could not generate trajectory angles file." << std::endl;
//     return result;
// }

// void PayloadController::printTrajectory()
// {
//     for (size_t i = 0; i < trajectory.poses.size(); i++)
//     {
//         const PlatformPose& pose = trajectory.poses[i];
//         std::cout
//             << "t=" << trajectory.times[i] << " ms  "
//             << "pos=["  << pose.position.transpose() << "]  "
//             << "ori=["  << pose.orientation.w() << " "
//                         << pose.orientation.x() << "i "
//                         << pose.orientation.y() << "j "
//                         << pose.orientation.z() << "k]  "
//             << "angles=[";
//         for (size_t j = 0; j < NUM_SERVOS; j++)
//         {
//             std::cout << trajectory.angles[i][j];
//             if (j < NUM_SERVOS - 1)
//                 std::cout << " ";
//         }
//         std::cout << "]" << std::endl;
//     }
// };

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