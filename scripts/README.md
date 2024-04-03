# Attentive Object Detection


## How to run

You need to follow these steps:

1. OpenPose
	- Location: IITICUBLAP235 (/users/emaiettini/mg_paper/gaze-estimation)
	- Run the container
	 ```
	 bash go
	 ```
2. Object detection
	- Location: IITICUBLAP156 (/home/icub/Users/emaiettini/detection_mutual_gaze)
	```
	bash execute_segmentation_mg.sh
	yarp rpc /detection/command:i
	>> load_model_detector shiva4_pringles_bleach_mustard
	```
3. Online visual target detection and Attentive object detection
	- Location: IITICUBLAP156 (/home/icub/Users/emaiettini/vtd)
	```
	bash go.sh
	```
	- Open the Yarpmanager and connect the modules using the `vtd_experiments.xml` application 
(/home/icub/Users/emaiettini/vtd/vtd_experiments.xml)

*NOTE: Adjust the input image port connected to the `/yarpOpenPose/image:i` based on using YARP dataplayer or camera*
