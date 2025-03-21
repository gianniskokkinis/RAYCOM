
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/llvm-13/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/llvm-13/lib/libLLVM.so'
import numpy as np
import sionna
from sionna.rt import Scene, Transmitter, Receiver, PlanarArray, load_scene
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.rt.antenna import iso_pattern
from trimesh import load_mesh


import argparse
import math


# ZMQ, PB
import zmq
import message_pb2

from commons import *

gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import node_from_map, node_name_from_input
import mitsuba as mi
import warnings
import time




gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')




# print("-----------------HERE PATHS -----------------")
# print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
# print("DRJIT_LIBLLVM_PATH:", os.environ.get('DRJIT_LIBLLVM_PATH'))


class SionnaEnv:
    
    
    def add_transmitter_to_scene(scene, updatePosition):
        tx = Transmitter(name="Laptop", position=updatePosition)
        scene.add(tx)
    
    def add_receiver_to_scene(scene, updatePosition):
        rx = Receiver(name="Router", position=updatePosition)
        scene.add(rx)

    def load_obj_from_file(scene, obj_file, updateMaterial="default"):
        #using trimesh lib 
        mesh = load_mesh(obj_file)
        for i, face in enumerate(mesh.faces):
            updateVertices = [mesh.vertices[idx] for idx in face] #transform vertices to load mesh vertices
            scene.add_object(name=f"object_{i}", vertices=updateVertices, material=updateMaterial)
    
    def compute_signal_effects(scene, tx_name="Laptop", rx_name="Router"):
        tx = scene.get(tx_name)
        rx = scene.get(rx_name)
        paths = scene.compute_paths(tx.position, rx.position)
        delays = paths.delays.numpy()
        #continue here
        

    

if __name__ == '__main__':  
    
   
    
    print("HELLO SIONNA!!!")