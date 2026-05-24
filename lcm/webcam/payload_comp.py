import lcm
from exlcm import payload_comp_msg_t, cam_msg_t

## PUBLISH FROM PAYLOAD COMP
# Create a message, fill in data fields and publish
def publish_payload_comp_msg(cam_enabled = False, exp_enabled = False):
    # Define msg
    msg = payload_comp_msg_t()
    msg.cam_enabled = cam_enabled
    msg.exp_enabled = exp_enabled

    # Intialise LCM
    lc = lcm.LCM()

    # Publish message to PAYLOAD_CAM channel
    lc.publish("PAYLOAD_CAM", msg.encode())

    print("Message published")


## RECEIVE FROM CAM
# Subscribe to and wait for msg from camera
def wait_for_cam_msg():
    # Define msg container
    received = {"msg": None}

    def payload_comp_handler(channel, data):
        received["msg"] = cam_msg_t.decode(data)
        print("Received message on channel \"%s\"" % channel)
        print("   cam_calib_complete     = %s" % str(received["msg"].cam_calib_complete))       
        print("   exp_complete     = %s" % str(received["msg"].exp_complete))
        print("")

    lc = lcm.LCM()
    sub = lc.subscribe("PAYLOAD_CAM", payload_comp_handler)

    # Wait for msg
    while received["msg"] is None:
        lc.handle()
    
    lc.unsubscribe(sub)

    return received["msg"]



#################################################################

# Publish camera calibration to start 
publish_payload_comp_msg(cam_enabled = True)

# Receive confirmation that camera calibration is complete
wait_for_cam_msg()

# Publish to start experiment
publish_payload_comp_msg(exp_enabled = True)

# Receive confirmation that experiment is complete
wait_for_cam_msg()
