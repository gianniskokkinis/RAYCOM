import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/llvm-13/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/llvm-13/lib/libLLVM.so'

# print("-----------------HERE PATHS -----------------")
# print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
# print("DRJIT_LIBLLVM_PATH:", os.environ.get('DRJIT_LIBLLVM_PATH'))

import argparse
import math

# ZMQ, PB
import zmq
import message_pb2

# Sionna
import os

from commons import *

gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sionna
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import node_from_map, node_name_from_input
import mitsuba as mi
import numpy as np
import warnings
import time
from trimesh import load_mesh

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.rt.antenna import iso_pattern
import matplotlib.pyplot as plt


class SionnaEnv:

    def __init__(self, updateScene, rt_calc_diffraction, rt_max_depth=5, rt_max_parallel_links=32, est_csi=True, VERBOSE=True):
        self.rt_calc_diffraction = rt_calc_diffraction
        self.rt_max_depth = rt_max_depth
        self.rt_max_parallel_links = rt_max_parallel_links
        self.est_csi = est_csi
        self.VERBOSE = VERBOSE
        self.node_info_dict = {}
        self.last_placed_nodes = [] # name of TX/RX placed during last channel computation
        self.pos_velo_cache = dict()
        #counter for packets here
        self.packet_counter = 0
        self.scene = updateScene
    
    
    def add_transmitter_to_scene(self, updatePosition):
        tx = Transmitter(name="Laptop", position=updatePosition)
        self.scene.add(tx)
    
    def add_receiver_to_scene(self, updatePosition):
        rx = Receiver(name="Router", position=updatePosition)
        self.scene.add(rx)

    def load_obj_from_file(self, obj_file, updateMaterial="default"):
        #using trimesh lib 
        mesh = load_mesh(obj_file)
        for i, face in enumerate(mesh.faces):
            updateVertices = [mesh.vertices[idx] for idx in face] #transform vertices to load mesh vertices
            self.scene.add_object(name=f"object_{i}", vertices=updateVertices, material=updateMaterial)
    
    def compute_signal_effects(self):
        tx = self.scene.get("Laptop")
        rx = self.scene.get("Router")
        paths = self.scene.compute_paths(tx.position, rx.position)
        delays = paths.delays.numpy()
        losses = paths.path_losses.numpy()
        print("Loss : ", losses)
        print("Delay : ", delays)
        

    def render_scene(self):
        
        #create Camera and initialize pars
        cameraPos= [2,2,2]
        lookAt = [0,0,0]
        cam = Camera(name="MainCamera", position=cameraPos)
        cam.look_at(lookAt)
        
        #add to scene 
        self.scene.add(cam)
        
        #take shot 
        img = self.scene.render(cam)
        
        # #display img 
        # plt.imshow(img)
        # plt.axis("off")
        # plt.title("CLI TOOL SCENE")
        # plt.show()
        

    

if __name__ == '__main__':  

    parser = argparse.ArgumentParser()
    parser.add_argument("--single_run", help="Whether not to terminate after single run", action='store_true')
    parser.add_argument("--rt_calc_diffraction", help="Calc diffraction in raytracing", action='store_true')
    parser.add_argument("--rt_max_depth", type=int, default=6, help="Calc diffraction in raytracing")
    parser.add_argument("--rt_max_parallel_links", type=int, default=4, help="Max no. of receivers")
    parser.add_argument("--est_csi", help="Whether to estimate complex CSI per OFDM subcarrier", action='store_true')
    parser.add_argument("--verbose", help="Whether to run in verbose mode", action='store_true')
    args = parser.parse_args()

    print("HELLO SIONNA!!!")

    

    #initialize scene
    filepath = "./../models/simple_room/simple_room.xml"
    scene = load_scene(filepath)
    env = SionnaEnv(scene, args.rt_calc_diffraction, args.rt_max_depth, args.rt_max_parallel_links, args.est_csi, VERBOSE=args.verbose)
    transPos = [0,0,2]
    env.add_transmitter_to_scene(transPos)
    receiverPos = [5,0,2]
    env.add_receiver_to_scene(receiverPos)
    # env.compute_signal_effects()
    
    # env.render_scene()
    
    