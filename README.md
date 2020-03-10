# VLinder AI

## 1. Presentation

VLinder AI is an intelligent road traffic and parking monitoring system. 

VLinder AI allows to provide interesting statistics by observing road traffic or parking in real time. 

VLinder is based on the [*Person Vehicle Bike detection crossroad 0078* model](https://docs.openvinotoolkit.org/2018_R5/_docs_Security_object_detection_crossroad_0078_caffe_desc_person_vehicle_bike_detection_crossroad_0078.html) of the [Intel Openvino tool](https://docs.openvinotoolkit.org/).

The information provided by VLinder is the number of people, the number of vehicles and the number of bikes in real time. And this information can then be used by the owners of the parking spaces (companies, shops, etc.) and also by the service. 
The owners of the parking spaces will use VLinder to find out for example if there is a space available for a vehicle or not, the number of bikes present, and also the number of people present in the parking at a given time and the number of people who have passed in one day.

 The traffic control services will use VLinder to be able to follow the state of the traffic in real time.
 
In the long term VLinder will also provide information about the level of congestion on a line, which is very interesting and important information for drivers and traffic control services to take the necessary measures. This ability to predict in real time the type or level of congestion will make VLinder an even more unique system.


***VLinder, Know where to drive and where to park quietly!***

## 2. How to test

### 1. Clone git repository and cd into the directory
```
git clone https://github.com/NathBangwa/VLinder-AI.git

cd VLinder-AI
```

### 2. Set up [virtualenv](https://virtualenv.pypa.io/en/stable/) with directory venv

```
virtualenv venv
```

### 3. Activate venv using

```
source venv/bin/activate
```

### 4. Install dependencies 
```
python -m pip install -r ./requirements/packagesPY.txt
```

### 5. User VLinder AI

##### 1. View args list
```
python app.py  -h

usage: Run inference on an input video [-h] [-i I] [-d D] [-p P]

optional arguments:
  -h, --help  show this help message and exit
  -i I        The location of the input file
  -d D        The device name, if not 'CPU'
  -p P        The device name, if not 'CPU'
```

##### 2. run inference
```
python run.py # default infer on ./testInputs/test_video.mp4

python app.py  -i ./testInputs/test_video.mp4 # infer to a specific input video

[Example]

(venv) root@60745d1cd77d:/home/workspace/MyProjectShowcase/VLinder-AI# python app.py 
True
INPUT VIDEO:  testInputs/test_video.mp4
OUTPUT VIDEO:  /home/workspace/MyProjectShowcase/VLinder-AI/outputs/OUT_test_video.mp4
[Processing]
..........................................................................................................................................................................................................[Finished]
(venv) root@60745d1cd77d:/home/workspace/MyProjectShowcase/VLinder-AI#
```

##### 3. View results

the output are storstored in folder ```./outputs/```
```
ls ./outputs/

[Example]

(venv) root@60745d1cd77d:/home/workspace/MyProjectShowcase/VLinder-AI# ls ./outputs/
OUT_test_video.mp4
```

## 6. Notice

As mentioned in the current version of VLinder does not yet support congestion detection.

For ease of testing, I put the vlinder project in my workspace on Udacity. Here are the details

```
[username on udacity: nathanbangwa@hotmail.com]
-My classroom
  -Intel® Edge AI Scholarship Foundation Course Nanodegree Program
    -Intel® Edge AI Foundation Course
      -Deploying an Edge App
        -Exercise: Server Communications
          -home
            -Workspace
              -MyProjectShowcase
                -VLinder-AI

```

