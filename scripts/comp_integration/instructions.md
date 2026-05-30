Generate lcm messages for c++ and python:

lcm-gen -p payload_msg_t.lcm

lcm-gen -x payload_msg_t.lcm

 \n Compile cpp:

g++ -std=c++14 payload_comp.cpp -o payload_comp $(pkg-config --cflags --libs lcm)


In one terminal:

python3 camera_master.py


Next terminal:

./payload_comp



(python implementation)

generate message with: lcm-gen -p payload_msg_t.lcm


in terminal: python3 cam.py (waits first, start here)

in terminal: python3 payload_comp.py


python3 lcm/webcam/camera_master.py

for pi:

* start camera first



to do:

* automate running both files? - save for with lincoln


