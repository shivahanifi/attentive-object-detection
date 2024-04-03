#!/bin/bash

xhost +

# Start Docker container
nvidia-docker run --rm -it --privileged --gpus all  --privileged -v /home/icub/Users/emaiettini/vtd:/RUN  -v /dev:/dev -e QT_X11_NO_MITSHM=1 -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --hostname dockerpc --network=host --pid=host visual_target_attention bash /RUN/run_vtd_aod.sh

