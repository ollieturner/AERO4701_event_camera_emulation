import lcm
from exlcm import example_t

# Create a message and fill in data fields
msg = example_t()
msg.timestamp = 0
msg.position = (1, 2, 3)
msg.orientation = (1, 0, 0, 0)
msg.ranges = range(15)
msg.num_ranges = len(msg.ranges)
msg.name = "example string"
msg.enabled = True

# Intialise LCM
lc = lcm.LCM()

# Publish message to EXAMPLE channel
lc.publish("EXAMPLE", msg.encode())