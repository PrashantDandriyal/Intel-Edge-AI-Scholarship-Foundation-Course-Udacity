###Run using Command: python feed_network.py -m models/human-pose-estimation-0001.xml###

import argparse
### TODO: Load the necessary libraries
from openvino.inference_engine import IECore, IENetwork, IEPlugin
import os

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml):
    ### TODO: Load the Inference Engine API
    ie = IECore()
    plugin = IEPlugin(device="CPU")
        
    ### TODO: Load IR files into their related class
    path_to_bin_file = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=path_to_bin_file)
 
    ### TODO: Add a CPU extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    ie.add_extension(extension_path=CPU_EXTENSION, device_name="CPU")
    exec_net = ie.load_network(net, "CPU")
    print(exec_net)
    
    ### TODO: Get the supported layers of the network
    #get_supported_layers(net)
    supp_layers = plugin.get_supported_layers(net)
    print("\nTHE SUPPORTED LAYERS ARE: " + str(supp_layers))
    
    ### TODO: Check for any unsupported layers, and let the user
    ###       know if anything is missing. Exit the program, if so.
    unsupp_layers = ie.query_network(network=net, device_name="CPU")
    print("\nTHE UNSUPPORTED LAYERS ARE: " + str(unsupp_layers))
    
    ### TODO: Load the network into the Inference Engine
    exec_net = plugin.load(network=net)
    
    print("IR successfully loaded into Inference Engine.")

    return


def main():
    args = get_args()
    load_to_IE(args.m)


if __name__ == "__main__":
    main()
