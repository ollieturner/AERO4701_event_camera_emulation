import lcm
from exlcm import example_t

# Callback function
def my_handler(channel, data):
    msg = example_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   timestamp   = %s" % str(msg.timestamp))
    print("   position    = %s" % str(msg.position))
    print("   orientation = %s" % str(msg.orientation))
    print("   ranges: %s" % str(msg.ranges))
    print("   name        = '%s'" % msg.name)
    print("   enabled     = %s" % str(msg.enabled))
    print("")

# Intialise LCM
lc = lcm.LCM()

# Subscribe to EXAMPLE channel
subscription = lc.subscribe("EXAMPLE", my_handler)

# Wait for message 
try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass