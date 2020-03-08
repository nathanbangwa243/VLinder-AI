import argparse
import cv2
import numpy as np
import socket
import json
from random import randint
from inference import Network
### TODO: Import any libraries for MQTT and FFmpeg
import paho.mqtt.client as mqtt
import sys

import os

INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = os.path.join(os.getcwd(), "intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so")
PVB_MODEL = os.path.join(os.getcwd(), "models/person-vehicle-bike-detection-crossroad-0078.xml")


CLASSES = ["person", "vehicule", "bike"]

TotalObjtectCounter = [0] * 3

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001  ### TODO: Set the Port for MQTT
MQTT_KEEPALIVE_INTERVAL = 60

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    p_desc = "Punish statistics, if not 'NO'"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    parser.add_argument("-d", help=d_desc, default='CPU')
    parser.add_argument("-p", help=d_desc, default='NO')
    args = parser.parse_args()

    return args

def addStatistics(frame, classIDs):
    global TotalObjtectCounter

    objectCounter = [classIDs.count(classID) for classID in range(3)]

    TotalObjtectCounter[0] += objectCounter[0]
    TotalObjtectCounter[1] += objectCounter[1]
    TotalObjtectCounter[2] += objectCounter[2]


    x = 50

    details = [
                ["Current Persons detected: ", objectCounter[0], (x, 50)],
                ["Current Vehicules detected: ", objectCounter[1], (x, 100)],
                ["Current Bikes detected: ", objectCounter[2], (x, 150)],

                ["Current Persons detected: ", TotalObjtectCounter[0], (x, 250)],
                ["Current Vehicules detected: ", TotalObjtectCounter[1], (x, 300)],
                ["Current Bikes detected: ", TotalObjtectCounter[2], (x, 350)]

            ]
    new_frame = None

    for text, nbObject, axis in details: 
        text += str(nbObject)

        new_frame = cv2.putText(frame,text, axis, cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
    
    return new_frame, objectCounter


def draw_boxes(frame, result, args, width, height):
    classes = []

    for box in result[0][0]: # Output shape 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            classID = box[1]

            classes.append(classID)

            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    return frame

def draw_masks(result, width, height):
    '''
    Draw semantic mask classes onto the frame.
    '''
    # Create a mask with color by class
    classes = cv2.resize(result[0].transpose((1,2,0)), (width,height), 
        interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    out_mask = classes * (255/20)
    
    # Stack the mask so FFmpeg understands it
    out_mask = np.dstack((out_mask, out_mask, out_mask))
    out_mask = np.uint8(out_mask)

    return out_mask, unique_classes


def get_class_names(class_nums):
    class_names= []
    for i in class_nums:
        class_names.append(CLASSES[int(i)])
    return class_names


def infer_on_video(args, model):
    ### TODO: Connect to the MQTT server
    client = mqtt.Client()
        
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    # Initialize the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(model, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # out writer video
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))


    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Perform inference on the frame
        plugin.async_inference(p_frame)

        # Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            # Draw the output mask onto the input
            out_frame, classes = draw_boxes(frame, result, args, width, height)
            class_names = get_class_names(classes)

            out_frame_local, stats = addStatistics(out_frame)

            # write out the frame
            out.write(out_frame_local)
            
            ### TODO: Send the class names and speed to the MQTT server
            ### Hint: The UI web server will check for a "class" and
            ### "speedometer" topic. Additionally, it expects "class_names"
            ### and "speed" as the json keys of the data, respectively.
            # publish class
            if args.p == "YES":
                speed = randint(50,70)
                dataClass = {
                    "class_names": class_names,
                }
                dataClass = json.dumps(dataClass)
                
                client.publish("class", dataClass)
                
                # publish speed
                dataSpeed = {
                    "speed": speed
                }
                
                dataSpeed = json.dumps(dataSpeed)
                
                client.publish("speedometer", dataSpeed)
        
        if args.p == "YES":
            ### TODO: Send frame to the ffmpeg server
            sys.stdout.buffer.write(out_frame)  
            sys.stdout.flush()
        
        

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()


def main():
    args = get_args()
    model = PVB_MODEL
    infer_on_video(args, model)


if __name__ == "__main__":
    main()
