import lcm
from exlcm import payload_comp_msg_t, cam_msg_t

# Subscribe to and wait for msg from payload computer
def wait_for_payload_comp_msg():
    # Define msg container
    received = {"msg": None}

    def cam_handler(channel, data):
        received["msg"] = payload_comp_msg_t.decode(data)
        print("Received message on channel \"%s\"" % channel)
        print("   cam_enabled     = %s" % str(received["msg"].cam_enabled))
        print("   exp_enabled     = %s" % str(received["msg"].exp_enabled))
        print("")

    lc = lcm.LCM()
    sub = lc.subscribe("PAYLOAD_CAM", cam_handler)

    # Wait for msg
    while received["msg"] is None:
        lc.handle()
    
    lc.unsubscribe(sub)

    return received["msg"]


# Create a message and fill in data fields
def publish_cam_msg(cam_calib_complete = False, exp_complete = False):
    # Define msg
    msg = cam_msg_t()
    msg.cam_calib_complete = cam_calib_complete
    msg.exp_complete = exp_complete

    # Intialise LCM
    lc = lcm.LCM()

    # Publish message to PAYLOAD_CAM channel
    lc.publish("PAYLOAD_CAM", msg.encode())

    print("Message published")