import lcm
from exlcm import payload_comp_msg_t

## PUBLISH FROM PAYLOAD COMP
# Create a message and fill in data fields
msg = payload_comp_msg_t()
msg.cam_enabled = True
msg.exp_enabled = True

# Intialise LCM
lc = lcm.LCM()

# Publish message to PAYLOAD_CAM channel
lc.publish("PAYLOAD_CAM", msg.encode())



## RECEIVE FROM CAM