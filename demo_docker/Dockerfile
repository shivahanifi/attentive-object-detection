#start from image passed by argument during build process. Usually it is an ubuntu image plus mesa library.
ARG START_IMG="none"
FROM $START_IMG

ENV DEBIAN_FRONTEND=noninteractive

##### Python #####
RUN apt-get update
RUN apt-get install -y python3-testresources

#### Terminator ####
RUN apt-get install -y terminator

##### Dependencies #####
RUN apt-get install -y libgoogle-glog-dev libboost-all-dev libhdf5-serial-dev libatlas-base-dev
RUN apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module
RUN apt-get install -y protobuf-compiler libprotobuf-dev

RUN apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev


# Install dependencies given in documentation in superbuild
# https://github.com/robotology/robotology-superbuild#linux
RUN mkdir -p /etc/bash_completion.d/ &&\
    apt-get update &&\
    apt-get install -y \
        # MISC
        bash-completion \
        qt5-default \
        git \
        cmake \
        cmake-curses-gui \ 
        build-essential \
        vim \
        #core
        coinor-libipopt-dev \ 
        libboost-system-dev \
        libboost-filesystem-dev \
        libboost-thread-dev \
        libopencv-dev \
        libode-dev \
        liblua5.1-dev \
        lua5.1 \ 
        libsdl1.2-dev \
        libxml2-dev\
        # YARP
        libace-dev \
        libeigen3-dev \
        libgsl0-dev \
        libedit-dev \
        libqcustomplot-dev \
        qtmultimedia5-dev \
        qtdeclarative5-dev \
        libqt5opengl5-dev \
        qttools5-dev \
        libtinyxml-dev \
        #yarp-gui
        qtbase5-dev \
        qtdeclarative5-dev \
        qtmultimedia5-dev \
        qml-module-qtquick2 \
        qml-module-qtquick-window2 \
        qml-module-qtmultimedia \
        qml-module-qtquick-dialogs \
        qml-module-qtquick-controls \
        qml-module-qt-labs-folderlistmodel \
        qml-module-qt-labs-settings 

#In order to install cmake 12.0 required by superbuild
RUN apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget &&\
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - &&\
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' &&\
    apt-get update &&\
    apt-get install -y kitware-archive-keyring &&\
    apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA &&\
    apt-get install -y cmake \
    &&\
    rm -rf /var/lib/apt/lists/* 


#### Install python bindings ####
RUN git config --global user.name icub
RUN git config --global user.email icub@iit.it


#Some definitions
ARG PROJECTS_DIR=/projects
ARG CMAKE_GENERATOR="Unix Makefiles"
ARG BUILD_TYPE=Release
ARG CMAKE_EXTRA_OPTIONS=-j2
ARG INSTALL_DIR="/usr/local"
ARG sbtag="Stable"


# Setup entrypoint
ARG ROBOTOLOGY_INITIALIZATION_FILE=/usr/local/bin/setup_robotology_tdd.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY setup.sh ${ROBOTOLOGY_INITIALIZATION_FILE}
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

RUN mkdir ${PROJECTS_DIR}

#### Install swig ####
RUN apt-get update && apt-get install -y automake byacc bison
RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/swig/swig.git &&\
    cd swig &&\
    git checkout tags/v4.0.2
RUN cd ${PROJECTS_DIR}/swig &&\
    ./autogen.sh &&\
    ./configure &&\
    make && make install

RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/robotology/robotology-superbuild.git &&\
    cd robotology-superbuild &&\
    git checkout tags/v2022.05.0 &&\
    mkdir build && cd build &&\
    cmake .. \
        -G "$CMAKE_GENERATOR" \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DNON_INTERACTIVE_BUILD:BOOL=ON \
        -DROBOTOLOGY_ENABLE_CORE:BOOL=ON \
        -DYCM_USE_DEPRECATED:BOOL=OFF \
        -DROBOTOLOGY_USES_GAZEBO=OFF \
        -DROBOTOLOGY_USES_PYTHON=ON \
        -DROBOTOLOGY_PROJECT_TAGS=${sbtag} \
        -DYCM_EP_INSTALL_DIR=${INSTALL_DIR} \
        &&\
    # Build all the projects
    cmake --build . --target update-all -- -j4 &&\
    cmake --build . -- ${CMAKE_EXTRA_OPTIONS}

ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3/dist-packages
ENV PYTHONPATH=$PYTHONPATH:${PROJECTS_DIR}/robotology-superbuild/build/src/YARP/lib/python3


######## START CUSTOM #########
#### Install python ####
RUN apt-get install -y python-dev
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
        python3.6 \
        python3-pip

RUN apt-get update && apt-get install -y \
        python3-distutils \
        python3-setuptools

RUN apt-get update

#### Install python packages ####
RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install --upgrade setuptools


### install also python3.5 #####
RUN apt-get update && apt-get install -y python3.5

# last conda with python3.6 (should see also python3.5)
RUN cd ${PROJECTS_DIR} &&\
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh &&\
    sha256sum Anaconda3-5.2.0-Linux-x86_64.sh
RUN cd ${PROJECTS_DIR} &&\
    bash Anaconda3-5.2.0-Linux-x86_64.sh -b

ENV PATH=/root/anaconda3/bin:${PATH}

RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/ejcgt/attention-target-detection.git &&\
    cd attention-target-detection &&\
    conda env create --name myenv-vtd -f environment.yml &&\
    pip install tensorflow

RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/shivahanifi/online-visual-target-detection.git &&\
    sh download_models.sh 
    


