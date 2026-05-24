import lcm
from exlcm import payload_comp_msg_t

## RECEIVE FROM PAYLOAD COMP
# Callback function
def my_handler(channel, data):
    msg = payload_comp_msg_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   cam_enabled     = %s" % str(msg.cam_enabled))
    print("   exp_enabled     = %s" % str(msg.exp_enabled))
    print("")


# Intialise LCM
lc = lcm.LCM()

# Subscribe to PAYLOAD_CAM channel
subscription = lc.subscribe("PAYLOAD_CAM", my_handler)

# Wait for message 
try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass


# ## PUBLISH FROM CAM
# # Create a message and fill in data fields
# msg = cam_msg_t()
# msg.exp_complete = True

# # Intialise LCM
# lc = lcm.LCM()

# # Publish message to PAYLOAD_CAM channel
# lc.publish("PAYLOAD_CAM", msg.encode())