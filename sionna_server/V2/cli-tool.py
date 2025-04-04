import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/llvm-13/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/llvm-13/lib/libLLVM.so'
# os.environ['MI_DEFAULT_VARIANT'] = 'llvm_ad_rgb'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

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


os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sionna
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import node_from_map, node_name_from_input
import mitsuba as mi
import numpy as np
import warnings
import time
from trimesh import load_mesh

# mi.set_variant('llvm_ad_rgb')
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)
# tf.get_logger().setLevel('ERROR')

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, AWGN
from sionna.rt.antenna import iso_pattern
from sionna.rt.scene_object import SceneObject
from sionna.utils.misc import *
from sionna.utils.metrics import *
from sionna.mapping import *
import matplotlib.pyplot as plt
import matplotlib as mat
from IPython.display import Image,display


    



class sionnaObject:

    """
    This is a sionna Object which add it to scene 

    """

    def __init__(self, updateObjName, updateObjFilePath, updateMaterial):
        self.objName = updateObjName
        self.objFilePath = updateObjFilePath
        self.material = updateMaterial
    
    def get_name(self):
        return self.objName
    
    def get_objFilePath(self):
        return self.objFilePath
    
    def get_material(self):
        return self.material
    



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
        self.new_object_id=1 #this is about new objects
        self.paths=[]



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
        print("------------ subcarrier_spacing ------------")
        print(subcarrier_spacing)
        fft_size = self.fft_size  # 64

        self.a, self.tau = 0, 0
        a_tau_set = False  

        # Compute propagation paths
        paths = self.scene.compute_paths(max_depth=self.rt_max_depth,
                                    method="fibonacci",
                                    num_samples=1e6,
                                    los=True,
                                    reflection=True,
                                    diffraction=self.rt_calc_diffraction,
                                    scattering=False)     

        self.paths = paths

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
                self.a, self.tau = los_path.cir()
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
                self.a = tf.concat([self.a, a_paths], axis=5)
                self.tau = tf.concat([self.tau, tau_paths], axis=3)
            else:
                self.a, self.tau = a_paths, tau_paths

        # Compute the frequencies of subcarriers and center around carrier frequency
        self.frequencies = subcarrier_frequencies(num_subcarriers=fft_size,
                                             subcarrier_spacing=subcarrier_spacing)

        # Compute the frequency response of the channel at frequencies
        self.h_freq = cir_to_ofdm_channel(frequencies=self.frequencies,
                                     a=self.a,
                                     tau=self.tau,
                                     normalize=False)
        

        # Disable normalization of delays for paths
        paths.normalize_delays = False      
        # Compute the channel impulse response for path
        a_paths, tau_paths = paths.cir()  

        # Calculate propagation delay
        lnk_delay = int(round(np.min(self.tau[self.tau >= 0] * 1e9), 0)) #ns

        # Calculate propagation loss
        lnk_loss = float(-10 * np.log10(tf.reduce_mean(tf.abs(self.h_freq) ** 2).numpy())) # Db

        #print channel info 

        print("Delay: ", lnk_delay)
        print("Loss: " , lnk_loss)
        # print("Frequencies ", self.frequencies.numpy())
        # print("Frequency Response : ", self.h_freq.numpy())
        # print("Impulse Response ")
        # print("amplitudes: ", self.a.numpy())
        # print("delays: ", self.tau.numpy())
        # print("PATHS: ", paths)

        
        # last_sim = simulation_time + (look_ahead - 1) * self.chan_coh_time_mode23
        # print("Calc channel finished:: LAH: Twin=%.6f -> %.6f" % (simulation_time/1e9, last_sim/1e9))


        

    def load_obj_from_file(self, sionnaObjects ,xml_file_path):
        
        
        
        checkList = []

        for obj in self.scene._scene_objects.values():
            checkList.append(obj._radio_material._name)
            print("Radio Material: ", obj._radio_material._name)

          

        print("---------- FILE ----------")
        fixLines = []
        f = open(xml_file_path, 'r')
        for line in f:

            if ("</scene>" in line):
                fixLines.append("<!-- Add New Items -->")
                
                #add materials
                for sionnaObj in sionnaObjects:
                    if (sionnaObj.get_material() not in checkList):
                        #test
                        print("self.scene.radio_materials : ", list(self.scene.radio_materials.keys()))    
                        print(f"{sionnaObj.get_material()} in {list(self.scene.radio_materials.keys())} : {sionnaObj.get_material() in list(self.scene.radio_materials.keys())}")
                        #end test
        
                        #check here if itu material from sionna list
                        if (sionnaObj.get_material() in list(self.scene.radio_materials.keys())):
                            checkList.append(sionnaObj.get_material())
                            print("ADD NEW MATERIAL : ",sionnaObj.get_material())
                            fixLines.append("\n")
                            fixLines.append(f'\t<bsdf type="twosided" id="mat-{sionnaObj.get_material()}">\n')
                            fixLines.append('\t\t<bsdf type="diffuse">\n')
                            fixLines.append('\t\t\t<rgb value="1 0 0" name="reflectance"/>\n')
                            fixLines.append('\t\t</bsdf>\n')
                            fixLines.append('\t</bsdf>\n')
                            
                        
                        else:
                            #create new material and set parameters
                            print("Not material in itu list ")
                            exit()
                    

                        #add radio material 
                        

                
                #add objects
                for sionnaObj in sionnaObjects:
                    fixLines.append("\n")
                    fixLines.append(f'\t<shape type="obj" id="mesh-{sionnaObj.get_name()}">\n')
                    fixLines.append(f'\t\t<string name="filename" value="{sionnaObj.get_objFilePath()}"/>\n')
                    fixLines.append(f'\t\t<boolean name="face_normals" value="true"/>\n')
                    fixLines.append(f'\t\t<ref id="mat-{sionnaObj.get_material()}" name="bsdf"/>\n')
                    fixLines.append(f'\t</shape>\n')

            fixLines.append(line)
        f.close()

        # for line in fixLines:
        #     print(line)

        
        get_path = xml_file_path.split('/')[0:-1]
        tempPath = ""
        for el in get_path:
            tempPath = tempPath + el + '/'
        tempPath += 'temp.xml'
        print(tempPath)
        
        
        f = open(tempPath, 'w')
        for line in fixLines:
            f.write(line)
        
        f.close()

       

        
        



        self.scene = load_scene(tempPath)
        self.scene.frequency = self.frequency
        self.scene.channel_bw = self.channel_bw
        self.scene.fft_size = self.fft_size

        
        os.remove(tempPath)


        
            
            



        # # self.scene._scene = mi.load_dict(temp_scene_dict)
        # self.scene._scene = mi.load_file(tempPath)
        # # os.remove(tempPath) #delete temp file

        # self.scene._scene_params = mi.traverse(self.scene._scene)

        # # Load the cameras
        # self.scene._load_cameras()

        # # Load the scene objects
        # self.scene._load_scene_objects()

        # # By default, no callable is used for radio materials
        # self.scene.radio_material_callable = None

        # # By default, no callable is used for scattering patterns
        # self.scene._scattering_pattern_callable = None

        # self.scene.scene_geometry_updated()
        
        
        
        



        #For Debbuging
        # print("------------- AFTER UPDATE -------------")
        # print("SHAPES : ", self.scene._scene.shapes)
        # print("OBJECTS : ", self.scene._scene_objects)
        # print("----------------------------------------")
        

    def simulate_digital_communication(self, ebno_db):
        
        
        #generate random bits 
        self.binary_source = BinarySource()


        self.modulations = {
            "pam" : {
                "type" : "pam" , 
                "num_bits" : 1
            },

            "4-pam" : {
                "type" : "pam" ,
                "num_bits" : 2
            },
            
            "16-qam" : {
                "type" : "qam",
                "num_bits" : 4
            },

            "64-qam" : {
                "type" : "qam",
                "num_bits" : 6
            }            
        }

        

        h_freq = self.h_freq.numpy()

        #test all modulations
        self.results = {}
        for modulation, modulation_params in self.modulations.items():
            
            num_bits = self.fft_size * modulation_params['num_bits']
            bits = self.binary_source([1, num_bits])
            
            #modulation
            if (modulation_params["type"] == "pam"):
                mapper = Mapper(constellation_type="pam", num_bits_per_symbol=modulation_params["num_bits"])
            else:
                mapper = Mapper(constellation_type="qam", num_bits_per_symbol=modulation_params["num_bits"])
            symbols = mapper(bits)

            #Using channel to sends
            symbols_ofdm = tf.signal.fft(tf.cast(symbols, tf.complex64))
            symbols_channel = symbols_ofdm * h_freq
            awgn_channel = AWGN()
            no = 10**(-ebno_db/10)
            y = awgn_channel((symbols_channel,no))

            #demodulation
            y_time = tf.signal.ifft(y)
            if (modulation_params["type"] == "pam"):
                demapper = Demapper(demapping_method="app", constellation_type="pam", num_bits_per_symbol=modulation_params["num_bits"])
            else:
                demapper = Demapper(demapping_method="app", constellation_type="qam", num_bits_per_symbol=modulation_params["num_bits"])
            
            llr = demapper([tf.expand_dims(y_time, axis=0), no])
            bits_hat = tf.cast(llr>0, tf.float32)

        
            #calculate Bit Error Rate 
            ber = compute_ber(bits, bits_hat)

            self.results[modulation] = {
                "ber" : ber.numpy(),
                "constellation" : symbols.numpy(),
                "received" : y_time.numpy()
            }

        



        




    
    def display_stats(self, material_display):
        
        mat.use("Qt5Agg")

        plt.figure(figsize=(15,10))
        
        plt.title(material_display)
        
        #display the frequency response Magnitude
        plt.subplot(2,2,1)
        display_frequencies = self.frequencies.numpy()
        #get the array
        display_h_freq = self.h_freq.numpy()[0][0][0][0][0][0][:]
        #convert the h_freq from imaginary to DB
        plt.plot(display_frequencies, 20*np.log10(np.abs(display_h_freq)), "b")
        plt.title("Frequency Response (Magnitude)")
        plt.xlabel("Frequencies (GHz)")
        plt.ylabel("Magnitude (DB)")
        plt.grid(True)

        #display the frequency response Phase 
        plt.subplot(2,2,2)
        plt.plot(display_frequencies, np.angle(display_h_freq), "r")
        plt.title("Frequency Response (Phase)")
        plt.xlabel("Frequencies (GHz)")
        plt.ylabel("phase (rad)")
        plt.grid(True)

        #display Impulse Response (Amplitudes and Delays)
        plt.subplot(2,2,3)
        amplitudes = self.a.numpy().flatten()
        display_amplitudes = np.abs(amplitudes)
        display_delays = self.tau.numpy().flatten()
        plt.stem(display_delays, display_amplitudes, linefmt="b-", markerfmt="bo", basefmt=' ')
        plt.title("Impulse Response (Power)")
        plt.xlabel("Delay [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        

        

        plt.tight_layout()
        plt.show()

        #plot Bit Error Rate 
        plt.figure(figsize=(15,10))
        plt.title("Bit Error Rate")
        
        #display BER    
        plt.subplot(2,4,1)    
        for modulation, result in self.results.items():
            plt.bar(modulation, result["ber"], alpha=0.6)
        plt.title("Bit Error Rate (BER)")
        plt.yscale("log")
        plt.grid(True)

        #display (4-PAM) 

        # Constellation
        plt.subplot(2,4,3)
        symbols = self.results["4-pam"]["constellation"]
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.3)
        plt.title("Modulated Symbols (4-pam)")
        plt.grid(True)

        # Receiver
        plt.subplot(2,4,4)
        symbols = self.results["4-pam"]["received"]
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.3)
        plt.title("Demodulated Symbols (4-pam)")
        plt.grid(True)


        #display (16-QAM)

        # Constellation
        plt.subplot(2,4,5)
        symbols = self.results["16-qam"]["constellation"]
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.3)
        plt.title("Modulated Symbols (16-pam)")
        plt.grid(True)

        # Receiver
        plt.subplot(2,4,6)
        symbols = self.results["16-qam"]["received"]
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.3)
        plt.title("Demodulated Symbols (16-pam)")
        plt.grid(True)


        #display (64-QAM)

        # Constellation
        plt.subplot(2,4,7)
        symbols = self.results["64-qam"]["constellation"]
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.3)
        plt.title("Modulated Symbols (64-pam)")
        plt.grid(True)

        # Receiver
        plt.subplot(2,4,8)
        symbols = self.results["64-qam"]["received"]
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.3)
        plt.title("Demodulated Symbols (64-pam)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()






            

        
        


    def preview_the_scene(self, updateResolution, updateCameraPosition, updateLookAt,updateFIlename):
        cameraPos = [updateCameraPosition[0], updateCameraPosition[1], updateCameraPosition[2]]
        lookAt = [updateLookAt[0], updateLookAt[1], updateLookAt[2]]
        set_camera = Camera(name="MainCamera", position=cameraPos)
        set_camera.look_at(lookAt)
        self.scene.add(set_camera)

        

        #test
        print("self.scene._scene.shapes() : ")
        print(self.scene._scene.shapes())
        #end test

        self.scene.render_to_file(
            camera = set_camera,
            filename=updateFIlename,
            resolution=updateResolution,
            paths = self.paths,
            show_paths = True,
            fov=60,
            show_devices=True,
            coverage_map = None,
            num_samples=1024
        )
        




def example1():

    #initialize scene
    filepath = "./../models/simple_room/simple_room.xml"
    scene = load_scene(filepath)
    # scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars)
    

    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64

    #setup enviroment and start simulation
    env = SionnaEnv(scene, frequency, bandwith, fft_size,  args.rt_calc_diffraction, args.rt_max_depth, args.rt_max_parallel_links, args.est_csi, VERBOSE=args.verbose)
    

    


    objectsToAdd = []

    
    
    #add objects 
    obj_file_path = "/home/user/Documents/Diplomatiki/objects_to_test/barrier_wall/barrier_wall.obj"
    sionObj = sionnaObject("barrier-wall",obj_file_path,"itu_brick")
    objectsToAdd.append(sionObj)

    obj_file_path = "/home/user/Documents/Diplomatiki/objects_to_test/barrier_wall/barrier_wall2.obj"
    sionObj = sionnaObject("barrier-wall-2",obj_file_path,"itu_brick")
    objectsToAdd.append(sionObj)


    obj_file_path = "/home/user/Documents/Diplomatiki/objects_to_test/barrier_wall/barrier_wall3.obj"
    sionObj = sionnaObject("barrier-wall-3",obj_file_path,"itu_brick")
    objectsToAdd.append(sionObj)

    obj_file_path = "/home/user/Documents/Diplomatiki/objects_to_test/barrier_wall/barrier_wall4.obj"
    sionObj = sionnaObject("barrier-wall-4",obj_file_path,"itu_brick")
    objectsToAdd.append(sionObj)
    
    
    env.load_obj_from_file(objectsToAdd, filepath)

        


    env.store_simulation_info()
    env.create_communication_link("Laptop", [1.5,2,1], "Router", [4.5,2,1])
    # env.create_communication_link("Laptop", [1.5,2,1], "Router", [1.25,2,1])
    env.calculate_channel_state()
    env.simulate_digital_communication(10)

    

    env.display_stats("itu_brick")
    
    # try:
    #     env.preview_the_scene([480,480],[-1.5,3,6],[4.5,2,1], "example1.jpg")
    #     print("!!! RENDER DONE !!!")
    # except Exception as e:
    #     print(e)
    
    

def example2():


    cameraPositions = { 
        "room1" : {
            "cameraPos" : [-5,-6,4],
            "lookAt" : [-2,1,3]
        },

        "room2" : { 
            "cameraPos" : [2,6,10] ,
            "lookAt" : [6,4,3]
        }
        
    }

    print("room1 -> cameraPos : ", cameraPositions["room1"]["cameraPos"])
    print("room1 -> lookAt : ", cameraPositions["room1"]["lookAt"])

    filepath = "./../models/futuristic_apartment/Futuristic_Apartment.xml"
    scene = load_scene(filepath)
    
    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64

    #setup enviroment and start simulation
    env = SionnaEnv(scene, frequency, bandwith, fft_size,  args.rt_calc_diffraction, args.rt_max_depth, args.rt_max_parallel_links, args.est_csi, VERBOSE=args.verbose)
    
    env.store_simulation_info()
    env.create_communication_link("Laptop", [6,4,3], "Router", [3,-7,3])
    env.calculate_channel_state()
    env.simulate_digital_communication(10)

    

    env.display_stats("itu_glass")

    
    
    try:
        resolution = [480,480]
        cameraPos = cameraPositions["room2"]["cameraPos"]
        lookAt = cameraPositions["room2"]["lookAt"]
        env.preview_the_scene(resolution, cameraPos, lookAt, "example1.jpg")
        print("!!! RENDER DONE !!!")
    except Exception as e:
        print(e)
    

def example3():

    cameraPositions = { 
        "place1" : {
            "cameraPos" : [-34,-132,100],
            "lookAt" : [-26,37,0]
        },

        "place2" : { 
            "cameraPos" : [-78,-165,30] ,
            "lookAt" : [-65,-143,0]
        },

        "place3" : { 
            "cameraPos" : [37,-268,114] ,
            "lookAt" : [-26,37,0]
        }
    }

    filepath = "./../models/Ioannina/Ioannina.xml"
    scene = load_scene(filepath)

    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64

    #setup enviroment and start simulation
    env = SionnaEnv(scene, frequency, bandwith, fft_size,  args.rt_calc_diffraction, args.rt_max_depth, args.rt_max_parallel_links, args.est_csi, VERBOSE=args.verbose)

    
    env.store_simulation_info()
    env.create_communication_link("Smartphone", [32,-93,30], "TelTower", [0,35,0])
    env.calculate_channel_state()
    env.simulate_digital_communication(10)

    

    env.display_stats("itu_glass")

    try:
        resolution = [1920,1080]
        cameraPos = cameraPositions["place3"]["cameraPos"]
        lookAt = cameraPositions["place3"]["lookAt"]
        env.preview_the_scene(resolution, cameraPos, lookAt, "example3.jpg")
        print("!!! RENDER DONE !!!")
    except Exception as e:
        print(e)
        

        
    


    
    
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

    example1()

    # example2() 

    # example3()   

    
    
    
    