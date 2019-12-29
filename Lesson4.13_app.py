#app.py
'''
These are the commands I used :
*Pre-requisites (read here: https://inteledgeaichallenge.slack.com/archives/DRKLV6VCZ/p1577608434001200)
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

tar -xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz


python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config
* Final Run command
python app.py -m frozen_inference_graph.xml -i test_video.mp4 -con 0.3 -col r

'''

import argparse
import cv2
import os
import numpy as np 

from inference import Network

INPUT_STREAM = "inTheEnd.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    con_desc = "The confidece thresholds used to draw bounding boxes"
    ###       2) The user choosing the color of the bounding boxes
    col_desc = "The color of the bounding boxes: 'b' : Blue, 'g' : Green, 'r' : Red, 'w' : White"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    
    optional.add_argument("-con", help=con_desc, default='0.5')
    optional.add_argument("-col", help=col_desc, default='b')
    args = parser.parse_args()

    return args

def get_colour(RBG):   #Follows the BGR Format
    if(RBG == 'b'):
        return (255, 0, 0)
    if(RBG == 'g'):
        return (0, 255, 0)
    if(RBG == 'r'):
        return (0, 0, 255)
    if(RBG == 'w'):
        return (0, 0, 0)

def draw_bounding_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        #print(box.shape)     # Dim: (7,)
        #print(type(box))     #NumPy array
        conf = box[2]
        if conf >= float(args.con):
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            #cv.Rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0) 
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), get_colour(args.col), 6) 
    return frame

def infer_on_video(args):
    ### TODO: Initialize the Inference Engine
    plugin = Network()
    
    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))      ## Method to get the dimensions of the video file
    height = int(cap.get(4))     ##

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        im = np.copy(frame)
        #Resize image
        b, c, h, w = plugin.get_input_shape()
        im = cv2.resize(im, (h,w))  #(W, H) AND Not (H,W) !!!
        #print(im.shape) #Prints (H, W, C)
        im = im.transpose((2,0,1))   #Gives (C, H, W)
        #Add another dimension to it
        im = im.reshape(1, 3, h, w)
        
        ### TODO: Perform inference on the frame
        plugin.async_inference(im)
        
        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()    #NumPy array of dim(1,1,100,7)
            #print(result.shape)
            ### TODO: Update the frame to include detected bounding boxes
            frame = draw_bounding_boxes(frame, result, args, width, height)
            # Write out the frame
            out.write(frame)
        # Break if escape key pressed
        if key_pressed == 27:    #ASCII for 'ESC' key
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
