This is the detailed explanation of how to use the object detection module combined with the VTD.

## Table of contents



## Fake Object detection output
In order to test with the already dumped data and without having to train the object detection module, we will create a fake output similar to the modules output. The below steps were followed to create the fake output:
1. Run `yarpserver --write` on the local machine
2. Run `yarpdataplayer --withExtraTimeCol 2` , and load the previously dumped data.

    - Note: I am using `seq0` from the first series of dumped data.
3. Start a `yarpview` from the terminal with a click to extract the pixel positions

    ```
    yarpview --name /test --out /click
    ```
4. Connect the `yarpview` and the dataplayer
    ```
    yarp connect /yarpOpenPose/propag:o /test
    ```
5. By clicking on the top-left and bottom-right corners of each of the objects the pixel positions will appear on the terminal.
6. Take one of the previous outputs of the object detection module `data.log` and replace the information you get.
    - NOTE: The information in the `data.log` are stored as: 

        ( Xtl Ytl Xbr Ybr Confidence detected_Label (Labels of all the objects that can be detected))
7. Since in the data we are using, everything is fixed and only person's gaze is changing so we can copy and paste the info we collected for one image, for all the images. 

    - NOTE: seq0 has 215 images. The related files for seq0 can be found [here]().