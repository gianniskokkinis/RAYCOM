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

    def __init__(self, updateScene, updateFrequency, updateBandwith, updateFft_size, update_fft_size ,rt_calc_diffraction, rt_max_depth=5, rt_max_parallel_links=32, est_csi=True, VERBOSE=True):
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
        self.frequency = updateFrequency
        self.channel_bw = updateBandwith
        self.fft_size = updateFft_size
        # Set scene parameters
        self.scene.frequency = updateFrequency
        self.scene.channel_bw = updateBandwith
        self.scene.fft_size = updateFft_size
        

    def store_simulation_info(self):

        # SISO mode only
        # Configure antenna array for all transmitters
        self.scene.tx_array = PlanarArray(num_rows=1,
                                     num_cols=1,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern=iso_pattern)    

            
        # Configure antenna array for all receivers
        self.scene.rx_array = PlanarArray(num_rows=1,
                                     num_cols=1,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern=iso_pattern)

        
                
        # If set to False, ray tracing will be done per antenna element (slower for large arrays)
        self.scene.synthetic_array = True
                

    
    
    
    def create_communication_link(self, tx_name, tx_position, rx_name , rx_position):
        #This method is about creating link between transmitter and receiver 
        self.tx = Transmitter(name=tx_name, position=tx_position)
        self.rx = Receiver(name=rx_name, position=rx_position)

        self.scene.add(self.tx)
        self.scene.add(self.rx)


    def calculate_channel_state(self):
        
        # WiFi parameters
        subcarrier_spacing = (self.channel_bw / self.fft_size)  # 312.5e3
        fft_size = self.fft_size  # 64

        a, tau = 0, 0
        a_tau_set = False  

        # Compute propagation paths
        paths = self.scene.compute_paths(max_depth=self.rt_max_depth,
                                    method="fibonacci",
                                    num_samples=1e6,
                                    los=True,
                                    reflection=True,
                                    diffraction=self.rt_calc_diffraction,
                                    scattering=False)     

        has_paths = bool(paths.types.numpy().size)
        has_los_path = np.any(paths.types.numpy()[0] == 0)

        # If no LOS path was found, check again with different compute_paths parameters
        # here calculating the amesi optiki epafi , Reflections and Diffraction
        if not has_los_path:
            los_path = self.scene.compute_paths(max_depth=0,
                                           method="fibonacci",
                                           num_samples=1e6,
                                           los=True,
                                           reflection=False,
                                           diffraction=False,
                                           scattering=False)

            has_los_path = bool(los_path.types.numpy().size)

            if not has_paths and not has_los_path:
                raise SystemExit(
                    "Error: Propagation loss and propagation delay cannot be calculated because no propagation paths were found. "
                    "Make sure that the nodes are not spatially separated in the 3D model and check the parameters of the compute_path() function.")
            if has_los_path:
                # Disable normalization of delays for LOS path
                los_path.normalize_delays = False
                # Compute the channel impulse response for LOS path
                a, tau = los_path.cir()
                a_tau_set = True        

        if has_paths:
            # Disable normalization of delays for paths
            paths.normalize_delays = False
            # Compute the channel impulse response for path
            a_paths, tau_paths = paths.cir()

            # Set a and tau
            # a -> the amplitude of the signal 
            # tay -> propagation delay
            if a_tau_set:
                a = tf.concat([a, a_paths], axis=5)
                tau = tf.concat([tau, tau_paths], axis=3)
            else:
                a, tau = a_paths, tau_paths

        # Compute the frequencies of subcarriers and center around carrier frequency
        frequencies = subcarrier_frequencies(num_subcarriers=fft_size,
                                             subcarrier_spacing=subcarrier_spacing)

        # Compute the frequency response of the channel at frequencies
        h_freq = cir_to_ofdm_channel(frequencies=frequencies,
                                     a=a,
                                     tau=tau,
                                     normalize=False)
        

        # Disable normalization of delays for paths
        paths.normalize_delays = False      
        # Compute the channel impulse response for path
        a_paths, tau_paths = paths.cir()  

        # Calculate propagation delay
        lnk_delay = int(round(np.min(tau[tau >= 0] * 1e9), 0)) #ns

        # Calculate propagation loss
        lnk_loss = float(-10 * np.log10(tf.reduce_mean(tf.abs(h_freq) ** 2).numpy())) # Db

        #print channel info 

        print("Delay: ", lnk_delay)
        print("Loss: " , lnk_loss)
        print("Frequency Response : ", h_freq.numpy())
        print("Impluse Response ")
        print("amplitudes: ", a.numpy())
        print("delays: ", tau.numpy())
        print("PATHS: ", paths)

        # last_sim = simulation_time + (look_ahead - 1) * self.chan_coh_time_mode23
        # print("Calc channel finished:: LAH: Twin=%.6f -> %.6f" % (simulation_time/1e9, last_sim/1e9))


        

    def load_obj_from_file(self, obj_file, updateMaterial="default"):
        #using trimesh lib 
        mesh = load_mesh(obj_file)
        for i, face in enumerate(mesh.faces):
            updateVertices = [mesh.vertices[idx] for idx in face] #transform vertices to load mesh vertices
            self.scene.add_object(name=f"object_{i}", vertices=updateVertices, material=updateMaterial)




    #this method is about running simulation
    def run_simulation(self, duration):
        pass
        
    
    def render_scene(self):
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection="3d")
        
        #plot Transmitter
        if self.tx:
            tx_pos = self.tx.position.numpy()
            ax.scatter(tx_pos[0], tx_pos[1], tx_pos[2], c="r", marker="^", s=100, label="Transmitter")

        #plot Receiver
        if self.rx:
            rx_pos = self.rx.position.numpy()
            ax.scatter(rx_pos[0], rx_pos[1], rx_pos[2], c="g", marker="o", s=100, label="Receiver")

        object_items = self.scene.objects.items()
        for ob_name, ob in object_items:
            if hasattr(ob, "position"):
                pos = ob.position.numpy()
                ax.scatter(pos[0], pos[1], pos[2], label=ob_name)

            # if ob_name not in ["tx", "rx"]:
            #     bb = ob.bounding_box()
            #     center = bb.center.numpy()
            #     size = bb.size.numpy()


                # x = [center[0]-size[0]/2, center[0]+size[0]/2]
                # y = [center[1]-size[1]/2, center[1]+size[1]/2]
                # z = [center[2]-size[2]/2, center[2]+size[2]/2]

                # for xi in x:
                #     for yi in y:
                #         ax.plot([xi,xi], [yi,yi], z, "b-", alpha=0.5)
                #     for zi in z:
                #         ax.plot([xi,xi], y, [zi,zi], "b-", alpha=0.5)
                    
                # for yi in y:
                #     for zi in z:
                #         ax.plot(x,[yi,yi],[zi,zi], "b-", alpha=0.5)

                # ax.text(center[0], center[1], center[2], ob_name, color="blue")
        
        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
        ax.set_zlabel("Z(m)")
        ax.set_title("Scene Graph")
        ax.legend()
        plt.tight_layout()
        plt.show()
    



    
    
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
    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64

    #setup enviroment and start simulation
    env = SionnaEnv(scene, frequency, bandwith, fft_size,  args.rt_calc_diffraction, args.rt_max_depth, args.rt_max_parallel_links, args.est_csi, VERBOSE=args.verbose)
    env.store_simulation_info()
    env.create_communication_link("Laptop", [5,0,2], "Router", [0,0,0])
    env.calculate_channel_state()
    env.render_scene()
    
    
    
    
    