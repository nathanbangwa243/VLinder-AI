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

## 7. See python, system, ram and hardware requirement

```
(venv) root@60745d1cd77d:/home/workspace/MyProjectShowcase/VLinder-AI# ls ./requirements/
cpu.txt  packagesPY.txt  ram.txt  system.txt

[cpu.txt]

(venv) root@60745d1cd77d:/home/workspace/MyProjectShowcase/VLinder-AI# cat ./requirements/cpu.txt 
processor       : 0
vendor_id       : GenuineIntel
cpu family      : 6
model           : 63
model name      : Intel(R) Xeon(R) CPU @ 2.30GHz
stepping        : 0
microcode       : 0x1
cpu MHz         : 2300.000
cache size      : 46080 KB
physical id     : 0
siblings        : 1
core id         : 0
cpu cores       : 1
apicid          : 0
initial apicid  : 0
fpu             : yes
fpu_exception   : yes
cpuid level     : 13
wp              : yes
flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm invpcid_single pti ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt arat md_clear arch_capabilities
bugs            : cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs itlb_multihit
bogomips        : 4600.00
clflush size    : 64
cache_alignment : 64
address sizes   : 46 bits physical, 48 bits virtual
power management:

[ram.txt]

(venv) root@60745d1cd77d:/home/workspace/MyProjectShowcase/VLinder-AI# cat ./requirements/ram.txt 
              total        used        free      shared  buff/cache   available
Mem:        3781572      388588      238968        4808     3154016     3083784
Swap:      16777212       19200    16758012

[system.txt]
(venv) root@60745d1cd77d:/home/workspace/MyProjectShowcase/VLinder-AI# cat ./requirements/system.txt 
Distributor ID: Ubuntu
Description:    Ubuntu 16.04.6 LTS
Release:        16.04
Codename:       xenial
```
