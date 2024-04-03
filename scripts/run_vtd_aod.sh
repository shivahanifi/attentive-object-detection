#!/bin/bash

terminator -u -T "VTD" -e "bash -c 'source activate myenv-vtd && cd /projects/online-visual-target-detection/src && python vtd_bbox_spt_series_input.py; exec bash'" &
terminator -u -T "AOD" -e "bash -c 'source activate myenv-vtd && cd /projects/attentive-object-detection/src && python aod.py; exec bash'"

