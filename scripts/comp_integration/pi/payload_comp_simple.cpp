#include <iostream>
#include <unistd.h>  // sleep()
#include <lcm/lcm-cpp.hpp>
#include "exlcm/payload_comp_msg_t.hpp"
#include "exlcm/cam_msg_t.hpp"


// g++ -std=c++14 payload_cam.cpp -o payload_cam $(pkg-config --cflags --libs lcm)

// Publish from Payload Comp

void publish_payload_comp_msg(lcm::LCM& lc,
                               bool cam_enabled = false,
                               bool exp_enabled = false)
{
    // Define msg
    exlcm::payload_comp_msg_t msg;
    msg.cam_enabled = cam_enabled;
    msg.exp_enabled = exp_enabled;

    // Publish msg
    lc.publish("PAYLOAD_CAM", &msg);
    std::cout << "Message published" << std::endl;
}


// Receive from Cam

// Handler class that stores the received message and signals completion
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

        std::cout << "Received message on channel \"" << channel << "\"" << std::endl;
        std::cout << "   cam_calib_complete = "
                  << (msg->cam_calib_complete ? "True" : "False") << std::endl;
        std::cout << "   exp_complete       = "
                  << (msg->exp_complete ? "True" : "False") << std::endl;
        std::cout << std::endl;
    }
};

// Subscribe and block until one message arrives, then return it
exlcm::cam_msg_t wait_for_cam_msg(lcm::LCM& lc)
{
    CamHandler handler;
    lcm::Subscription* sub = lc.subscribe("PAYLOAD_CAM",
                                           &CamHandler::handleMessage,
                                           &handler);

    while (!handler.received)
        lc.handle();

    lc.unsubscribe(sub);
    return handler.last_msg;
}


// Main

int main(int argc, char** argv)
{
    lcm::LCM lc;
    if (!lc.good()) {
        std::cerr << "Failed to initialise LCM" << std::endl;
        return 1;
    }

    // Publish to start camera calibration
    publish_payload_comp_msg(lc, /*cam_enabled=*/true);

    // Wait for confirmation that camera calibration is complete
    wait_for_cam_msg(lc);

    sleep(2);

    // Publish to start experiment
    publish_payload_comp_msg(lc, /*cam_enabled=*/false, /*exp_enabled=*/true);

    // Wait for confirmation that experiment is complete
    wait_for_cam_msg(lc);

    return 0;
}