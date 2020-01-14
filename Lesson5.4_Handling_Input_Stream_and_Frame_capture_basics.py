#This one was very important for learning the baics of taking in image/video streams 
#I got the error in 'frame = cv2.resize(frame,(100,100))" :
#   ..."error: (-215:Assertion failed) !ssize.empty() in function 'resize'"
#Actually it was caused due to lack of frames to resize. ONce the image had been processed, the capture didn't close
#...(I thought isOpened() would handle it. but No) so I had to use the "ret, frame = cap.read(), if not ret: break" in line 45-46 

import argparse
import cv2
import numpy as np

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Handle an input stream")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input (image or video) file OR 'webcam' for accessing the webcam"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc)
    args = parser.parse_args()

    return args


def capture_stream(args):
    ### TODO: Handle image, video or webcam
    video_flag = False
    
    #If video file
    if args.i.endswith('.mp4'):  
        video_flag = True         #Doing this for separating methods to write a video file later
        out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (100,100)) #Note: do this outside the loop, its to be only once
    #If image file
    elif args.i.endswith('jpg'):
        pass
    elif args.i == 'webcam':
        args.i = 0
    
    ### TODO: Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)                #Note
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:                  #Note
            break
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Re-size the frame to 100x100
        frame = cv2.resize(frame,(100,100))
        
        ### TODO: Add Canny Edge Detection to the frame, 
        ###       with min & max values of 100 and 200
        frame_with_edges = cv2.Canny(frame,100,200)
        
        ### Make sure to use np.dstack after to make a 3-channel image
        #img_array = np.array(frame_with_edges)
        #print(img_array.shape)
        img_array_1 = np.dstack((frame_with_edges, frame_with_edges, frame_with_edges)) 
        
        ### TODO: Write out the frame, depending on image or video
        if video_flag: 
            #Create a video writer 
            out.write(img_array_1)
        else:
            cv2.imwrite('output_img.jpg', img_array_1)
        
        ### TODO: Close the stream and any windows at the end of the application
        # Break if escape key pressed
        if key_pressed == 27:
            break
    
    #Destroy all the writers (out for video)
    if video_flag:
        out.release()
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = get_args()
    capture_stream(args)


if __name__ == "__main__":
    main()

