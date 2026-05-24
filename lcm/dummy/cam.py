import lcm
from exlcm import payload_comp_msg_t, cam_msg_t
import time

## RECEIVE FROM PAYLOAD COMP
# Subscribe to and wait for msg from payload computer
def wait_for_payload_comp_msg():
    # Define msg container
    received = {"msg": None}

    def cam_handler(channel, data):
        received["msg"] = payload_comp_msg_t.decode(data)
        # print("Received message on channel \"%s\"" % channel)
        # print("   cam_enabled     = %s" % str(received["msg"].cam_enabled))
        # print("   exp_enabled     = %s" % str(received["msg"].exp_enabled))
        # print("")

    lc = lcm.LCM()
    sub = lc.subscribe("PAYLOAD_CAM", cam_handler)

    # Wait for msg
    while received["msg"] is None:
        lc.handle()
    
    lc.unsubscribe(sub)

    return received["msg"]

# Stop program untill cam msg received
msg = wait_for_payload_comp_msg()

# Print out msg
print("Continue program")
print(msg.cam_enabled)
print(msg.exp_enabled)
print("\n")

# Buffer wait time
time.sleep(3)

## PUBLISH FROM CAM
# Create a message and fill in data fields
def publish_cam_msg():
    # Define msg
    msg = cam_msg_t()
    msg.exp_complete = True

    # Intialise LCM
    lc = lcm.LCM()

    # Publish message to PAYLOAD_CAM channel
    lc.publish("PAYLOAD_CAM", msg.encode())

    print("Message published")

publish_cam_msg()