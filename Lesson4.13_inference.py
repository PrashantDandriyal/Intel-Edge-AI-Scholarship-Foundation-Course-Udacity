'''
Notes: 

* iter() and next() in Python
eg.
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit)) 
print(next(myit))
print(next(myit))

> apple
> banana
> cherry
_____________________________________________________________

Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)   
        '''
        - where network is An ExecutableNetwork object
        - This object can be used to get the input mapping and output mapping
        - here, mapping refers to the numpy arrays mapped to the layer names, eg. input_blob, output_blob
        '''
        

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        #print(self.network.inputs[self.input_blob].shape)
        return self.network.inputs[self.input_blob].shape


    def async_inference(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        ### TODO: Start asynchronous inference
        infer_request_handle = self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        #print(infer_request_handle)     #InferRequest object
        return 


    def wait(self):
        '''
        Checks the status of the inference request.
        
        ### TODO: Wait for the async request to be complete
        '''
        status = self.exec_network.requests[0].wait(-1)   #Pauses the execution right here 
        return status


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        ### TODO: Return the outputs of the network from the output_blob
        #print(self.exec_network.requests[0].outputs.keys()) #Gives 'DetectionOutput' as its only key
        return self.exec_network.requests[0].outputs[self.output_blob]                  
