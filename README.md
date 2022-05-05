# depthai_publisher
EGB349 OAK-D Lite DepthAI Publisher

## Step 1 - Clone Repository

```
cd [ros_ws]/src/
git clone https://github.com/dennisbrar/depthai_publisher.git
```

## Step 2 - Build Packages
```
cd ../
```

#### Source ROS Noetic packages

```
catkin_make
```

## Step 3 - Run the 

#### Terminal 1:
```
roscore
```

#### Terminal 2:
```
rosrun depthai_publisher dai_publisher
```

#### Terminal 3:
```
rosrun depthai_publisher aruco_subscriber
```

#### Terminal 4:
```
rqt_image_view
```
OR
```
rviz
```