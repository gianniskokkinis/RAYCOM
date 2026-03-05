import os

# -----------------------------------------------------------------------------
# Mitsuba/DrJIT (Sionna RT) runtime setup
# -----------------------------------------------------------------------------
# Sionna RT selects the Mitsuba variant in `sionna.rt.__init__`:
# - GPU available  -> 'cuda_ad_rgb'
# - CPU only       -> 'llvm_ad_rgb'  (requires LLVM runtime)
# On Windows, DrJIT needs to locate `LLVM-C.dll`. If LLVM is installed system-wide
# (common default path), we set DRJIT_LIBLLVM_PATH automatically so imports work.
if os.name == "nt" and "DRJIT_LIBLLVM_PATH" not in os.environ:
    _llvm_candidates = [
        r"C:\Program Files\LLVM\bin",
        r"C:\Program Files (x86)\LLVM\bin",
    ]
    for _d in _llvm_candidates:
        _dll = os.path.join(_d, "LLVM-C.dll")
        if os.path.exists(_dll):
            # DrJIT expects a path that helps it locate LLVM. On Windows, pointing
            # directly to `LLVM-C.dll` is the most reliable option.
            os.environ["DRJIT_LIBLLVM_PATH"] = _dll
            break

# Keep this for safety; Sionna will still call `mi.set_variant(...)` internally.
os.environ.setdefault("MI_DEFAULT_VARIANT", "llvm_ad_rgb")

# Linux-only Qt hint (harmless elsewhere, but avoid setting it on Windows)
if os.name != "nt":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse



os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sionna
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import node_from_map, node_name_from_input
import mitsuba as mi
import numpy as np
import warnings
import time
import trimesh

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
from sionna.rt.antenna import iso_pattern,tr38901_pattern
from sionna.rt.scene_object import SceneObject
from sionna.rt.radio_material import RadioMaterial
from sionna.utils.misc import *
from sionna.utils.metrics import *
from sionna.utils.plotting import *
from sionna.mapping import *
import matplotlib.pyplot as plt
import matplotlib as mat
from PyQt5.QtWidgets import QScrollArea, QLabel, QVBoxLayout, QWidget
import time 
import json
from PIL import Image

    



class sionnaObject:

    """
    This is a sionna Object which add it to scene 

    """

    def __init__(self, updateObjName, updateObjFilePath, updatePos):
        self.objName = updateObjName
        self.objFilePath = updateObjFilePath
        self.hasCustomMaterial = False
        self.position = updatePos
        
    
    def get_name(self):
        return self.objName
    
    def get_objFilePath(self):
        return self.objFilePath
    
    def get_material(self):
        return self.material

    def set_sionna_material(self, updateMaterialName):
        self.material = updateMaterialName

    def create_custom_material(self, material_name, 
                                material_relative_permittivity, 
                                material_conductivity,
                                material_scattering_coefficient,
                                material_xpd_coefficient,
                                material_scattering_pattern,
                                material_frequency_update_callback,
                                setColor
                            ):
        self.hasCustomMaterial = True
        #create material and set it to sionnaObj
        self.material = RadioMaterial( name = material_name,
                                        relative_permittivity = material_relative_permittivity,
                                        conductivity = material_conductivity,
                                        scattering_coefficient = material_scattering_coefficient,
                                        xpd_coefficient = material_xpd_coefficient,
                                        scattering_pattern = material_scattering_pattern,
                                        frequency_update_callback = material_frequency_update_callback,
                                    )
        self.color = setColor

    def get_custom_material_name(self):
        return self.material._name

    def set_position(self, updatePOS):
        self.position = updatePOS

    def get_position(self):
        return self.position
    
    def set_color(self, rgbCode):
        self.color = rgbCode
    
    def get_color(self):
        return self.color

    def get_formatted_color(self):
        return f"{self.color[0]} {self.color[1]} {self.color[2]}"
    


class SionnaEnv:

    def __init__(self, updateScene, updateFrequency, updateBandwith, updateFft_size, update_fft_size ,rt_calc_diffraction, rt_max_depth=5, rt_max_parallel_links=32, est_csi=True, VERBOSE=True):
        self.rt_calc_diffraction = rt_calc_diffraction
        self.rt_max_depth = rt_max_depth
        self.rt_max_parallel_links = rt_max_parallel_links
        self.est_csi = est_csi
        self.VERBOSE = VERBOSE
        self.scene = updateScene
        self.frequency = updateFrequency
        self.channel_bw = updateBandwith
        self.fft_size = updateFft_size
        # Set scene parameters
        self.scene.frequency = updateFrequency
        self.scene.channel_bw = updateBandwith
        self.scene.fft_size = updateFft_size
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
        self.scene.rx_array = PlanarArray(num_rows=5,
                                     num_cols=5,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern=iso_pattern)

        
                
        # If set to False, ray tracing will be done per antenna element (slower for large arrays)
        self.scene.synthetic_array = True
                

    
    
    
    # Πρόσθεσα το rx_orientation=[0,0,0] στα ορίσματα (με default τιμή για να μην σπάει ο κώδικας αν δεν το δώσεις)
    def create_communication_link(self, tx_name, tx_position, rx_name , rx_position, rx_orientation=[0,0,0]):
        #This method is about creating link between transmitter and receiver 
        self.tx = Transmitter(name=tx_name, position=tx_position)
        
        # Τώρα το rx_orientation υπάρχει και μπορεί να χρησιμοποιηθεί εδώ
        self.rx = Receiver(name=rx_name, position=rx_position, orientation=rx_orientation)

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
            # tay ->  delay
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

        # Calculate delay
        self.lnk_delay = int(round(np.min(self.tau[self.tau >= 0] * 1e9), 0)) #ns

        # Calculate loss
        self.lnk_loss = float(-10 * np.log10(tf.reduce_mean(tf.abs(self.h_freq) ** 2).numpy())) # Db


        #print channel info 
        print(f"Delay: {self.lnk_delay} sec")
        print(f"Loss: {self.lnk_loss} sec")
        print(f"Bandwith: {self.scene.channel_bw} Hz")
        


        #calculate RMS Delay Spread 
        if len(self.a.shape) >= 6:
            a_for_stats = self.a[:, :, 0, :, 0, :, :] 
        else:
            a_for_stats = self.a # case SISO 
            
        a_np = np.abs(a_for_stats.numpy().flatten()) # |ak|
        tau_np = self.tau.numpy().flatten() # |tk|
        
        # Now a_np and tau_np will both have a size of 37 (or however many paths were found)
        # and the calculation will be performed correctly.
        
        self.mean_tau = np.sum((a_np**2) * tau_np) / np.sum(a_np**2) # mean tau 
        self.rms_delay_spread = np.sqrt(np.sum((a_np**2) * (tau_np - self.mean_tau)**2) / np.sum(a_np**2) ) #total rms


        #calculate coherence bandwith
        self.coherence_bandwith = 1 / (5 * self.rms_delay_spread)

        print(f"Coherence Bandwith: {self.coherence_bandwith} Hz")
        
            
        
    

    def convert_stl_to_obj(self,stl_file_path):
        stl_mesh = trimesh.load(stl_file_path)
        changeFileName = stl_file_path.split("/")[-1].split(".")[0]+".obj"
        fixPath = stl_file_path.split("/")
        fixPath.pop()
        fixPath.append(changeFileName)
        destination = "/".join(fixPath)
        stl_mesh.export(destination)
        return destination

        

    def load_obj_from_file(self, sionnaObjects ,xml_file_path):
        
        
        
        #Get materials from the current scene to skip creating them again
        checkList = []
        for scObj in self.scene._scene_objects.values():
            checkList.append(scObj.radio_material._name)
          
        
        fixLines = [] #to add specific lines to xml 
        sionnaObjectsWithCustomMaterials = [] #objects with custom materials 
        customMaterialsCheck = [] #to avoid double initialize custom materials to scene 
        convertedPaths = [] #contains the path of *.stl which we are gonna convert these
        f = open(xml_file_path, 'r')
        for line in f:

            if ("</scene>" in line):
                fixLines.append("<!-- Add New Items -->")
                
                #add materials
                for sionnaObj in sionnaObjects:
                    if (sionnaObj.get_material() not in checkList):
                        #check here if itu material from sionna list
                        if (sionnaObj.get_material() in list(self.scene.radio_materials.keys())):
                            checkList.append(sionnaObj.get_material())
                            fixLines.append("\n")
                            fixLines.append(f'\t<bsdf type="twosided" id="mat-{sionnaObj.get_material()}">\n')
                            fixLines.append('\t\t<bsdf type="diffuse">\n')
                            fixLines.append('\t\t\t<rgb value="1 0 0" name="reflectance"/>\n')
                            fixLines.append('\t\t</bsdf>\n')
                            fixLines.append('\t</bsdf>\n')
                        else:
                            if (sionnaObj.get_material().name not in customMaterialsCheck):
                                #set custom metarial
                                customMaterialsCheck.append(sionnaObj.get_material().name)
                                fixLines.append("\n")
                                fixLines.append(f'\t<bsdf type="diffuse" id="custom-visual-{sionnaObj.get_material().name}">\n')
                                fixLines.append(f'\t\t<rgb name="reflectance" value="{sionnaObj.get_formatted_color()}"/>\n')
                                fixLines.append("\t</bsdf>")
                                fixLines.append(f'\n')
                            
                            

                       
                        

                
                #add objects
                for sionnaObj in sionnaObjects:
                    if (sionnaObj.hasCustomMaterial == False):
                        #ITU-Material
                        fixLines.append("\n")
                        fixLines.append(f'\t<shape type="obj" id="mesh-{sionnaObj.get_name()}">\n')

                        #print("extension: ", sionnaObj.get_objFilePath().split("/")[-1].split(".")[1]) #TEST
                        #case need convert 
                        if (sionnaObj.get_objFilePath().split("/")[-1].split(".")[1] != "obj"):
                            path = self.convert_stl_to_obj(sionnaObj.get_objFilePath())
                            convertedPaths.append(path)
                        else:
                            path = sionnaObj.get_objFilePath()
            
                        fixLines.append(f'\t\t<string name="filename" value="{path}"/>\n')
                        fixLines.append(f'\t\t<boolean name="face_normals" value="true"/>\n')

                        #set positions
                        x = sionnaObj.position[0]
                        y = sionnaObj.position[1]
                        z = sionnaObj.position[2]
                        fixLines.append(f'\t\t<transform name="to_world">\n')
                        fixLines.append(f'\t\t\t<translate x="{x}" y="{y}" z="{z}"/>\n')
                        fixLines.append(f"\t\t</transform>\n")
                        

                        fixLines.append(f'\t\t<ref id="mat-{sionnaObj.get_material()}" name="bsdf"/>\n')
                        fixLines.append(f'\t</shape>\n')
                    else:
                        #CUstom material
                        sionnaObjectsWithCustomMaterials.append(sionnaObj)
                        fixLines.append("\n")
                        fixLines.append(f'\t<shape type="obj" id="mesh-{sionnaObj.get_name()}">\n')

                        #case need convert
                        #print("extension: ", sionnaObj.get_objFilePath().split("/")[-1].split(".")[1]) #TEST
                        if (sionnaObj.get_objFilePath().split("/")[-1].split(".")[1] != "obj"):
                            path = self.convert_stl_to_obj(sionnaObj.get_objFilePath())
                            convertedPaths.append(path)
                        else:
                            path = sionnaObj.get_objFilePath()

                        fixLines.append(f'\t\t<string name="filename" value="{path}"/>\n')
                        fixLines.append(f'\t\t<boolean name="face_normals" value="true"/>\n')

                        #set positions
                        x = sionnaObj.position[0]
                        y = sionnaObj.position[1]
                        z = sionnaObj.position[2]
                        fixLines.append(f'\t\t<transform name="to_world">\n')
                        fixLines.append(f'\t\t\t<translate x="{x}" y="{y}" z="{z}"/>\n')
                        fixLines.append(f"\t\t</transform>\n")

                        fixLines.append(f'\t\t<ref id="custom-visual-{sionnaObj.get_material().name}"/>\n')
                        fixLines.append(f'\t</shape>\n')
                        
                        

            fixLines.append(line)
        f.close()

        

        
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

        #assign the material from sionna custom objects materials 
        for customMatObj in sionnaObjectsWithCustomMaterials:

            if (customMatObj.get_custom_material_name() not in self.scene.radio_materials.keys()):
                #set material
                self.scene.get(customMatObj.get_name()).radio_material=customMatObj.material
            else:
                #just perform the name of the material
                self.scene.get(customMatObj.get_name()).radio_material=customMatObj.get_custom_material_name()
    

        #delete the converted files 
        for delPath in convertedPaths:
            os.remove(delPath)

        
        
        

    def simulate_digital_communication(self,update_batch_size, update_iter):

        self.snr_values = []
        
        #initialize snr 
        self.snr_values = np.linspace(0,30,31)
        

        
        
        print("self.snr_values :",self.snr_values)

        #generate random bits 
        self.binary_source = BinarySource()

        #modulations 
        self.modulations = {
            "pam" : {
                "type" : "pam" , 
                "num_bits" : 1,
                "ber" : [],
                "bler" : [],
                "ser" : [],
                "bit_errors" : []
            },

            "4-pam" : {
                "type" : "pam" ,
                "num_bits" : 2,
                "ber" : [],
                "bler" : [],
                "ser" : [],
                "bit_errors" : []
            },
            
            "16-qam" : {
                "type" : "qam",
                "num_bits" : 4,
                "ber" : [],
                "bler" : [],
                "ser" : [],
                "bit_errors" : []
            },

            "64-qam" : {
                "type" : "qam",
                "num_bits" : 6,
                "ber" : [],
                "bler" : [],
                "ser" : [],
                "bit_errors" : []
            }            
        }
        
        #get frequency from the channel
        h_freq = self.h_freq.numpy()

       

        #initialize array for bit_errors results
        temp_bit_errors_results = []
        



        #function which is gonna called via PlotBer.simulate(...)
        def simulation(batch_size,ebno_db):

            
            
            
            #initialize bits
            num_symbols = self.fft_size
            total_bits = num_symbols * modulation_params["num_bits"]
            
            
           


            #create transmitted bits
            bits = self.binary_source([batch_size, total_bits])
            

            #modulation - Mapping
            if (modulation_params["type"] == "pam"):
                mapper = Mapper(constellation_type="pam", num_bits_per_symbol=modulation_params["num_bits"])
            else:
                mapper = Mapper(constellation_type="qam", num_bits_per_symbol=modulation_params["num_bits"])
            
            symbols = mapper(bits)

            
            #normalization symbols 
            symbols_power = tf.cast(tf.reduce_mean(tf.abs(symbols)**2),tf.complex64)
            symbols = symbols / tf.sqrt(symbols_power)
            

            #channel effects
            symbols_ofdm = tf.signal.fft(tf.cast(symbols, tf.complex64)) #convert symbols to Frequencies Field using FFT    
            symbols_channel = symbols_ofdm * h_freq 
            awgn_channel = AWGN() #add the AWGN model for realistic noise 
            no = 1.0 / ((10**(ebno_db/10)))
            y = awgn_channel((symbols_channel,no)) #add the noise to channel(signal)

            #demodulation - demapping
            y_time = tf.signal.ifft(y) * np.sqrt(self.fft_size) #convert from frequency to time
            if (modulation_params["type"] == "pam"):
                demapper = Demapper(demapping_method="app", constellation_type="pam", num_bits_per_symbol=modulation_params["num_bits"])
            else:
                demapper = Demapper(demapping_method="app", constellation_type="qam", num_bits_per_symbol=modulation_params["num_bits"])
                    
            llr = demapper([y_time,no]) #calculate the llr
            bits_hat = tf.cast(llr>0, tf.float32) #get the received bits

                
            #calculate Bitwise Mutual Information 
            bit_errors = count_errors(bits, bits_hat)
            temp_bit_errors_results.append(np.mean(bit_errors))

            
            return bits,bits_hat

        
        

        

        

        
        plot_ber = {}
        for modulation in self.modulations:
            plot_ber[modulation] = PlotBER(title=f"Bit/Block Error Rate ({modulation})")

        #for every modulation run simulation
        for modulation, modulation_params in self.modulations.items():
            
            print(f"\n\n{modulation} modulation: ")

            

            for snr in self.snr_values:

                #clear table for next experiment
                temp_bit_errors_results.clear()


                ber, bler = plot_ber[modulation].simulate(
                    mc_fun=simulation,
                    ebno_dbs=[snr],
                    batch_size=update_batch_size,
                    max_mc_iter=update_iter,
                    legend = f"{modulation} at {snr} dB",
                    add_ber=True,
                    add_bler=True,
                    show_fig=False,
                )

                # print("BER: ",ber)
                # print("BLER: ",bler)
                modulation_params["ber"].append(ber)
                modulation_params["bler"].append(bler)
                modulation_params["bit_errors"].append(np.mean(temp_bit_errors_results))
                
                
                
                
            
           

    def generate_heatmap(self):
        # Get the channel response (frequency range)
        # h_freq shape: [1, 1, 25, 1, 1, fft_size, 1] (assuming 1 Tx, 1 Rx object, SISO TX, 5x5 RX)
        h = self.h_freq.numpy() 
        
        # Elimination of dimensions that are 1
        # Now we have the shape [25, fft_size] -> [number_of_antennas, frequencies]
        h_squeezed = np.squeeze(h) 
        
        # We select, for example, the central subcarrier (frequency) for display.
        # Or we can take the average of all frequencies.
        # Let's take the average power across all subcarriers for starters.
        
        # Magnitude Calculation
        magnitude = np.mean(np.abs(h_squeezed), axis=1) # We take the average of the subcarriers. Result: 25 values
        
        # Phase calculation - a little more complex because the phase changes with frequency
        # Let's take the phase of the central subcarrier
        center_subcarrier = h_squeezed.shape[1] // 2
        phase = np.angle(h_squeezed[:, center_subcarrier]) # Result: 25 values
        
        # Reshape into a 5x5 grid to make it an image
        magnitude_map = magnitude.reshape(5, 5)
        phase_map = phase.reshape(5, 5)
        
        return magnitude_map, phase_map



    def plot_radio_image(self):
        mag_map, phase_map = self.generate_heatmap()
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(mag_map, cmap='viridis', interpolation='nearest') # 'nearest' to show pixels, 'bicubic' for smoother results
        plt.colorbar(label='Magnitude')
        plt.title('RF Magnitude Map (5x5)')
        
        plt.subplot(1, 2, 2)
        plt.imshow(phase_map, cmap='twilight', interpolation='nearest')
        plt.colorbar(label='Phase (rad)')
        plt.title('RF Phase Map (5x5)')
        
        plt.show()
        
        # Here you will save the images for pix2pix later
        # plt.imsave("rf_input.png", mag_map, cmap='viridis')
        
    
    def save_radio_heatmap(self, filename):
        
        # collect data 
        mag_map, phase_map = self.generate_heatmap()
        
        # Saving to file
        # We use 'plt.imsave' which creates an image without axes and margins.
        # 'cmap' gives the color.
        plt.imsave(filename, mag_map, cmap='viridis') 
        
        print(f"--> Saved Heatmap image: {filename}")


    def save_dataset_pair(self, pair_id, folder="dataset"):

        # --- CREATE FOLDERS ---
        path_A = os.path.join(folder, "A") # RF Heatmaps
        path_B = os.path.join(folder, "B") # Optical Images

        if not os.path.exists(path_A): os.makedirs(path_A)
        if not os.path.exists(path_B): os.makedirs(path_B)

        # --- 2. RF HEATMAP (INPUT) ---
        # We generate the Heatmap from the signal received by the Receiver.
        mag_map, phase_map = self.generate_heatmap()
        
        # Το αποθηκεύουμε ως εικόνα
        plt.imsave(os.path.join(path_A, f"{pair_id}_A.png"), mag_map, cmap='viridis')

        # --- OPTICAL IMAGE (TARGET) ---
        
        # We take the coordinates
        rx_loc = self.rx.position.numpy().flatten() # Πού είναι το Router
        tx_loc = self.tx.position.numpy().flatten() # Πού είναι το Laptop

        # We calculate the direction from the router to the laptop.
        direction = tx_loc - rx_loc
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance # Κανονικοποίηση (γίνεται μονάδα)
        else:
            direction = np.array([1.0, 0.0, 0.0])

        # Πιο ασφαλής θέση κάμερας:
        # 1. Πιο μακριά από τον router (π.χ. 1.0 m)
        # 2. Λίγο πιο ψηλά (π.χ. +0.3 στο z) ώστε να είμαστε σίγουρα εκτός mesh
        camera_pos = rx_loc + (direction * 1.0)
        camera_pos[2] += 0.3

        # Ρυθμίζουμε την κάμερα
        cam_name = "dataset_cam"
        # Αν υπάρχει ήδη παλιά κάμερα, την σβήνουμε
        if cam_name in self.scene.cameras:
            self.scene.remove(cam_name)
            
        camera = Camera(cam_name, position=camera_pos.tolist())
        camera.look_at(tx_loc.tolist()) # Η κάμερα κοιτάει το Laptop
        self.scene.add(camera)
        
        # Κάνουμε Render
        filename_B = os.path.join(path_B, f"{pair_id}_B.png")
        self.scene.render_to_file(
            camera=cam_name, 
            filename=filename_B, 
            resolution=(256, 256),           # Μικρή ανάλυση για γρήγορο dataset
            paths=self.paths,                # χρήση των ίδιων paths με το preview
            show_paths=False,
            fov=60,
            show_devices=False,
            coverage_map=None,
            num_samples=256                  # λιγότερα samples για ταχύτητα
        )

        # DEBUG: Check if the final image is truly black or just dark
        try:
            img = Image.open(filename_B).convert("RGB")
            arr = np.array(img)
            max_val = arr.max()
            mean_val = arr.mean()
            print(f"[DEBUG] B image stats for pair {pair_id}: max={max_val}, mean={mean_val}")
            if max_val == 0:
                print(f"[DEBUG] Pair {pair_id}: image is completely black")
        except Exception as e:
            print(f"[DEBUG] Could not analyze B image for pair {pair_id}: {e}")
        
        # Clear
        self.scene.remove(cam_name)

        print(f"Saved Pair {pair_id} (Rx moved to y={rx_loc[1]:.2f})")
    


    def display_stats(self):
        
        mat.use("Qt5Agg")
        
        # Καθαρισμός παλιών figures για να αποφύγουμε τα warnings
        plt.close('all')

        plt.figure(figsize=(15,10))
        plt.suptitle("Channel Informations") # Χρήση suptitle για κεντρικό τίτλο
        
        # --- 1. Frequency Response (Magnitude) ---
        plt.subplot(2,2,1)
        display_frequencies = self.frequencies.numpy()
        
        # ΑΣΦΑΛΗΣ ΤΡΟΠΟΣ ΕΞΑΓΩΓΗΣ:
        # Κάνουμε τον πίνακα flat (μονοκόμματο) και παίρνουμε τα πρώτα 'fft_size' στοιχεία.
        # Αυτό αντιστοιχεί αυτόματα στην 1η κεραία, 1ο ζεύγος Tx-Rx, ανεξαρτήτως διαστάσεων.
        h_flat = self.h_freq.numpy().flatten()
        display_h_freq = h_flat[:self.fft_size] 

        #convert the h_freq from imaginary to DB
        plt.plot(display_frequencies, 20*np.log10(np.abs(display_h_freq)), "b")
        plt.title("Frequency Response (Magnitude - 1st Ant.)")
        plt.xlabel("Frequencies (GHz)")
        plt.ylabel("Magnitude (DB)")
        plt.grid(True)

        # --- 2. Frequency Response (Phase) ---
        plt.subplot(2,2,2)
        plt.plot(display_frequencies, np.angle(display_h_freq), "r")
        plt.title("Frequency Response (Phase - 1st Ant.)")
        plt.xlabel("Frequencies (GHz)")
        plt.ylabel("phase (rad)")
        plt.grid(True)

        # --- 2. Frequency Response (Phase) ---
        plt.subplot(2,2,2)
        plt.plot(display_frequencies, np.angle(display_h_freq), "r")
        plt.title("Frequency Response (Phase - 1st Ant.)")
        plt.xlabel("Frequencies (GHz)")
        plt.ylabel("phase (rad)")
        plt.grid(True)

        # --- 3. Impulse Response (Amplitudes and Delays) ---
        plt.subplot(2,2,3)
        
        # ΔΙΟΡΘΩΣΗ: Επιλογή της 1ης κεραίας για τα Paths
        # self.a shape: [batch, rx, rx_ant, tx, tx_ant, paths, steps]
        if len(self.a.shape) >= 6:
             # Παίρνουμε την 1η κεραία (index 0 στο rx_ant)
            a_vis = self.a[:, :, 0, :, 0, :, :]
        else:
            a_vis = self.a
            
        amplitudes = a_vis.numpy().flatten()
        display_amplitudes = np.abs(amplitudes)
        display_delays = self.tau.numpy().flatten() #convert to nanoseconds
        
        # Τώρα τα μεγέθη είναι σωστά (π.χ. 37 vs 37) και δεν θα πετάξει error
        plt.stem(display_delays, display_amplitudes, linefmt="b-", markerfmt="bo", basefmt=' ')
        plt.title("Impulse Response (Power - 1st Ant.)")
        plt.xlabel("Delay [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)

        if (len(display_delays)>0):#case small values
            max_delay = np.max(display_delays) #take the max limit 
            # Προστασία για διαίρεση με το μηδέν ή πολύ μικρά νούμερα
            if max_delay == 0:
                max_delay = 1e-9
            min_limit = -0.01*max_delay
            max_limit = 1.01*max_delay
            plt.xlim([min_limit, max_limit])
        

        #legend box with objects and materials 
        displayList = []
        for obj in self.scene._scene_objects.values():
            displayList.append(f'{obj._name} : {obj._radio_material._name}')
        plt.subplot(2,2,4)
        plt.axis("off")
        info_text = "\n".join(displayList)

        #simple window
        if (len(displayList)<25):    
            plt.text(
                0.5,
                0.5,
                info_text,
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(
                    facecolor = "white",
                    edgecolor = "black",
                    boxstyle = "round,pad=0.5",
                    alpha=0.8
                )
            )
        #scroll pane window 
        else:
            
            scroll_area = QScrollArea()
            scroll_content = QWidget()
            layout = QVBoxLayout(scroll_content)

            for obj in displayList:
                layout.addWidget(QLabel(obj))
            
            scroll_area.setWidget(scroll_content)
            scroll_area.setWidgetResizable(True)
            scroll_area.show()
            

        plt.tight_layout()
        plt.show()

        #display channel info (ΣΕ ΞΕΧΩΡΙΣΤΟ ΠΑΡΑΘΥΡΟ ΓΙΑ ΝΑ ΜΗΝ ΠΕΦΤΕΙ ΠΑΝΩ ΣΤΑ ΑΛΛΑ)
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        info_text = "---- Channel Informations ----\n"+"Delay: "+str(self.lnk_delay)+"sec\n"+"Loss: "+str(self.lnk_loss)+"sec\n"+"Bandwith: "+str(self.scene.channel_bw)+"Hz\n"+"Coherence Bandwith: "+str(self.coherence_bandwith)+"Hz\n"
        plt.text(
                0.5,
                0.5,
                info_text,
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(
                    facecolor = "white",
                    edgecolor = "black",
                    boxstyle = "round,pad=0.5",
                    alpha=0.8
                )
            )
        plt.tight_layout()
        plt.show()

        #plot errors ration
        plt.figure(figsize=(15,10))
        plt.title("Error Ration")



        # """
        # FOR BER 
        # """
        # #display PAM BER 
        # plt.subplot(2,4,1)
        # ber_display = self.modulations["pam"]["ber"]
        # plt.semilogy(self.snr_values, ber_display, "-o")
        # plt.title("PAM modulation BER")
        # plt.xlabel("Eb/No (dB)")
        # plt.ylabel("BER")
        # plt.grid(True)

        # #display PAM BLER
        # plt.subplot(2,4,2)
        # ber_display = self.modulations["pam"]["bler"]
        # plt.semilogy(self.snr_values, ber_display, "-o")
        # plt.title("PAM modulation BLER")
        # plt.xlabel("Eb/No (dB)")
        # plt.ylabel("BLER")
        # plt.grid(True)

        # #display 4-PAM BER 
        # plt.subplot(2,4,3)
        # ber_display = self.modulations["4-pam"]["ber"]
        # plt.semilogy(self.snr_values, ber_display, "-o")
        # plt.title("4-PAM modulation BER")
        # plt.xlabel("Eb/No (dB)")
        # plt.ylabel("BER")
        # plt.grid(True)

     
        # #display 4-PAM BLER
        # plt.subplot(2,4,4)
        # ber_display = self.modulations["4-pam"]["bler"]
        # plt.semilogy(self.snr_values, ber_display, "-o")
        # plt.title("4-PAM modulation BLER")
        # plt.xlabel("Eb/No (dB)")
        # plt.ylabel("BLER")
        # plt.grid(True)



        # #display 16-QAM BER 
        # plt.subplot(2,4,5)
        # ber_display = self.modulations["16-qam"]["ber"]
        # plt.semilogy(self.snr_values, ber_display, "-o")
        # plt.title("16-QAM modulation BER")
        # plt.xlabel("Eb/No (dB)")
        # plt.ylabel("BER")
        # plt.grid(True)

  
        # #display 16-QAM BLER
        # plt.subplot(2,4,6)
        # ber_display = self.modulations["16-qam"]["bler"]
        # plt.semilogy(self.snr_values, ber_display, "-o")
        # plt.title("16-QAM modulation BLER")
        # plt.xlabel("Eb/No (dB)")
        # plt.ylabel("BLER")
        # plt.grid(True)


        # #display 64-QAM BER 
        # plt.subplot(2,4,7)
        # ber_display = self.modulations["64-qam"]["ber"]
        # plt.semilogy(self.snr_values, ber_display, "-o")
        # plt.title("64-QAM modulation BER")
        # plt.xlabel("Eb/No (dB)")
        # plt.ylabel("BER")
        # plt.grid(True)


        # #display 64-QAM BLER
        # plt.subplot(2,4,8)
        # ber_display = self.modulations["64-qam"]["bler"]
        # plt.semilogy(self.snr_values, ber_display, "-o")
        # plt.title("64-QAM modulation BLER")
        # plt.xlabel("Eb/No (dB)")
        # plt.ylabel("BLER")
        # plt.grid(True)

        # plt.tight_layout()
        # plt.show()
        

        # #plot Bits Error 
        # plt.figure(figsize=(15,10))
        # plt.title("Bits Error")
        # pos = 1
        # for mod in self.modulations.keys():
        #     plt.subplot(2,4,pos)
        #     display_values = self.modulations[mod]["bit_errors"]
        #     plt.semilogy(self.snr_values, display_values, "-o")
        #     plt.title(f"{mod} modulation Bits Error")
        #     plt.xlabel("Eb/No (dB)")
        #     plt.ylabel("Bits Error")
        #     plt.grid(True)
        #     pos+=1
        # plt.tight_layout()
        # plt.show()
        
        
        






            

        
        


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



        




def example1(isRender):


    materialsToCheck = {
        "mirror" : {
            "material_name" : "mirror",
            "material_relative_permittivity" : 1.0,
            "material_conductivity" : 3.8e7,
            "material_scattering_coefficient" : 0.0,
            "material_xpd_coefficient" : 0.0,
            "material_scattering_pattern" : None,
            "material_frequency_update_callback" : None
        },
        "mercury_wall" : {
            "material_name" : "mercury_wall",
            "material_relative_permittivity" : 1.0,
            "material_conductivity" : 1e6,
            "material_scattering_coefficient" : 0.0,
            "material_xpd_coefficient" : 0.0,
            "material_scattering_pattern" : None,
            "material_frequency_update_callback" : None
        },
        "elevator" : {
            "material_name" : "elevator",
            "material_relative_permittivity" : 1.0,
            "material_conductivity" : 1.4e6, #anojeidwto atsali
            "material_scattering_coefficient" : 0.05, #elafros anwmali epifaneia
            "material_xpd_coefficient" : 0.1,
            "material_scattering_pattern" : None,
            "material_frequency_update_callback" : None
        },
        "drywall" : {
            "material_name" : "drywall",
            "material_relative_permittivity" : 2.5,
            "material_conductivity" : 0.01, #anojeidwto atsali
            "material_scattering_coefficient" : 0.05, #elafros anwmali epifaneia
            "material_xpd_coefficient" : 0.1,
            "material_scattering_pattern" : None,
            "material_frequency_update_callback" : None
        },
          

        
       

    }

    #initialize scene
    filepath = "./scenes/ex1/simple_room.xml"
    scene = load_scene(filepath)
    # scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars)
    

    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64

    #setup enviroment and start simulation
    env = SionnaEnv(scene, frequency, bandwith, fft_size,  False, 6, 4, False, VERBOSE=False)
    

    

    
    objectsToAdd = []


    
    #add objects 
    obj_file_path = "./obj/ex1/barrier_wall.stl"
    sionObj = sionnaObject("barrier-wall",obj_file_path,[3,2.00469,0.422312])

    # sionObj.set_sionna_material("itu_brick")
    sionObj.create_custom_material(
        material_name=materialsToCheck["mirror"]["material_name"],
        material_relative_permittivity=materialsToCheck["mirror"]["material_relative_permittivity"],
        material_conductivity=materialsToCheck["mirror"]["material_conductivity"],
        material_scattering_coefficient=materialsToCheck["mirror"]["material_scattering_coefficient"],
        material_xpd_coefficient=materialsToCheck["mirror"]["material_xpd_coefficient"],
        material_scattering_pattern=materialsToCheck["mirror"]["material_scattering_pattern"],
        material_frequency_update_callback=materialsToCheck["mirror"]["material_frequency_update_callback"],
        setColor=[1.0, 0.0, 0.0]
    )

    objectsToAdd.append(sionObj)

    obj_file_path = "./obj/ex1/barrier_wall2.stl"
    sionObj = sionnaObject("barrier-wall-2",obj_file_path,[3,2.00469,1.90484])
    # sionObj.set_sionna_material("itu_brick")
    sionObj.create_custom_material(
        material_name=materialsToCheck["mercury_wall"]["material_name"],
        material_relative_permittivity=materialsToCheck["mercury_wall"]["material_relative_permittivity"],
        material_conductivity=materialsToCheck["mercury_wall"]["material_conductivity"],
        material_scattering_coefficient=materialsToCheck["mercury_wall"]["material_scattering_coefficient"],
        material_xpd_coefficient=materialsToCheck["mercury_wall"]["material_xpd_coefficient"],
        material_scattering_pattern=materialsToCheck["mercury_wall"]["material_scattering_pattern"],
        material_frequency_update_callback=materialsToCheck["mercury_wall"]["material_frequency_update_callback"],
        setColor=[0.0 , 1.0, 0.0]
    )



    objectsToAdd.append(sionObj)


    obj_file_path = "./obj/ex1/barrier_wall3.stl"
    sionObj = sionnaObject("barrier-wall-3",obj_file_path, [3,0.936904,1.2271])
    # sionObj.set_sionna_material("itu_brick")
    sionObj.create_custom_material(
        material_name=materialsToCheck["elevator"]["material_name"],
        material_relative_permittivity=materialsToCheck["elevator"]["material_relative_permittivity"],
        material_conductivity=materialsToCheck["elevator"]["material_conductivity"],
        material_scattering_coefficient=materialsToCheck["elevator"]["material_scattering_coefficient"],
        material_xpd_coefficient=materialsToCheck["elevator"]["material_xpd_coefficient"],
        material_scattering_pattern=materialsToCheck["elevator"]["material_scattering_pattern"],
        material_frequency_update_callback=materialsToCheck["elevator"]["material_frequency_update_callback"],
        setColor=[0.0, 0.0, 1.0]
    )



    objectsToAdd.append(sionObj)

    obj_file_path = "./obj/ex1/barrier_wall4.stl"
    sionObj = sionnaObject("barrier-wall-4",obj_file_path, [3,3.34588,1.2271])
    # sionObj.set_sionna_material("itu_brick")
    sionObj.create_custom_material(
        material_name=materialsToCheck["drywall"]["material_name"],
        material_relative_permittivity=materialsToCheck["drywall"]["material_relative_permittivity"],
        material_conductivity=materialsToCheck["drywall"]["material_conductivity"],
        material_scattering_coefficient=materialsToCheck["drywall"]["material_scattering_coefficient"],
        material_xpd_coefficient=materialsToCheck["drywall"]["material_xpd_coefficient"],
        material_scattering_pattern=materialsToCheck["drywall"]["material_scattering_pattern"],
        material_frequency_update_callback=materialsToCheck["drywall"]["material_frequency_update_callback"],
        setColor=[1.0, 1.0, 0.0]
    )




    objectsToAdd.append(sionObj)
    
    
    env.load_obj_from_file(objectsToAdd, filepath)

        


    env.store_simulation_info()
    env.create_communication_link("Laptop", [1.5,2,1], "Router", [4.5,2,1])
    env.calculate_channel_state()
    # env.simulate_digital_communication(1000,10)

    

    env.display_stats()

    

    if (isRender):
        #render 
        try:
            env.preview_the_scene([1280,720],[-1.5,3,6],[4.5,2,1], "example1.jpg")
            print("!!! RENDER DONE !!!")
        except Exception as e:
            print(e)

    #for color map 
    env.generate_heatmap()
    env.save_radio_heatmap("my_heatmap.png")
    
    

def example2(isRender):


    cameraPositions = { 
        "room1" : {
            "cameraPos" : [-6,-3,4],
            "lookAt" : [3,-8.25,3]
        },

        "room2" : { 
            "cameraPos" : [4.3,6,3] ,
            "lookAt" : [4,1,3]
        }
        
    }

    #mirror materials
    materialsToCheck = {
        "mirror" : {
            "material_name" : "mirror",
            "material_relative_permittivity" : 1.0,
            "material_conductivity" : 3.8e7,
            "material_scattering_coefficient" : 0.0,
            "material_xpd_coefficient" : 0.0,
            "material_scattering_pattern" : None,
            "material_frequency_update_callback" : None
        }
    }

    # print("room1 -> cameraPos : ", cameraPositions["room1"]["cameraPos"])
    # print("room1 -> lookAt : ", cameraPositions["room1"]["lookAt"])

    filepath = "./scenes/ex2/Futuristic_Apartment.xml"
    scene = load_scene(filepath)
    
    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64

    #setup enviroment and start simulation
    env = SionnaEnv(scene, frequency, bandwith, fft_size,  False, 6, 4, False, VERBOSE=False)
    
    #add mirror to the scene
    obj_file_path = "./obj/ex2/Mirror.obj"
    sionObj = sionnaObject("Mirror",obj_file_path, [-1,-7.6,3])
    sionObj.create_custom_material(
        material_name=materialsToCheck["mirror"]["material_name"],
        material_relative_permittivity=materialsToCheck["mirror"]["material_relative_permittivity"],
        material_conductivity=materialsToCheck["mirror"]["material_conductivity"],
        material_scattering_coefficient=materialsToCheck["mirror"]["material_scattering_coefficient"],
        material_xpd_coefficient=materialsToCheck["mirror"]["material_xpd_coefficient"],
        material_scattering_pattern=materialsToCheck["mirror"]["material_scattering_pattern"],
        material_frequency_update_callback=materialsToCheck["mirror"]["material_frequency_update_callback"],
        setColor=[0.0, 0.0, 1.0]
    )

    sionnaObjects = [sionObj]
    env.load_obj_from_file(sionnaObjects, filepath)

    

    env.store_simulation_info()
    env.create_communication_link("Laptop", [6,4,3], "Router", [5,-7,3])
    env.calculate_channel_state()
    env.simulate_digital_communication(1000,10)

    

    env.display_stats()

    if (isRender):
        #rendering 
        try:
            resolution = [1280,720]
            
            #render first room
            cameraPos = cameraPositions["room1"]["cameraPos"]
            lookAt = cameraPositions["room1"]["lookAt"]
            env.preview_the_scene(resolution, cameraPos, lookAt, "example2_1.jpg")
            


            print("!!! RENDER DONE !!!")
        except Exception as e:
            print(e)

 

   
        
    

def example3(isRender,place):

    cameraPositions = { 
        "place1" : {
            "cameraPos" : [34,-132,51.5],
            "lookAt" : [-26,37,0]
        },

        "place2" : { 
            "cameraPos" : [42.5,-380,62.7] ,
            "lookAt" : [-26,37,0]
        },

        "place3" : { 
            "cameraPos" : [-175,-150,20] ,
            "lookAt" : [-26,37,0]
        }
    }

    smartphonePositions = {

        "pos1" : [-8,-31.5,3.6],
        "pos2" : [-52.1,-237.1,3.6],
        "pos3" : [-141.3,-115.1,5.5]
    }

    filepath = "./scenes/ex3/Ioannina.xml"
    scene = load_scene(filepath)

    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64

    """
    POSITION 1
    
    """

    print("---------------------------------- POS OUTSIDE OTE -------------")
    if (place==1):

        #setup enviroment and start simulation
        env1 = SionnaEnv(scene, frequency, bandwith, fft_size,  False, 6, 4, False, VERBOSE=False)

        #add the antenna
        obj_file_path = "./obj/ex3/Antenna.obj"
        sionObj = sionnaObject("Antenna",obj_file_path, [0,0,0])
        sionObj.set_sionna_material("itu_metal")
        objectsToAdd = [sionObj]
        env1.load_obj_from_file(objectsToAdd, filepath)
        
        env1.store_simulation_info()
        env1.create_communication_link("Smartphone", smartphonePositions["pos1"], "TelTower", [0,10,50])
        env1.calculate_channel_state()
        env1.simulate_digital_communication(1000,10)

        

        env1.display_stats()

        if (isRender):
            try:
                resolution = [1280,720]
                cameraPos = cameraPositions["place1"]["cameraPos"]
                lookAt = cameraPositions["place1"]["lookAt"]
                env1.preview_the_scene(resolution, cameraPos, lookAt, "example3_1.jpg")
                print("!!! RENDER DONE !!!")
            except Exception as e:
                print(e)
                
    elif (place==2): 
    
        """
        POSITION 2
        
        """

        print("---------------------------------- POS OUTSIDE NOMARXIA -------------")

        #setup enviroment and start simulation
        scene = load_scene(filepath)
        env2 = SionnaEnv(scene, frequency, bandwith, fft_size,  False, 6, 4, False, VERBOSE=False)

        #add the antenna
        obj_file_path = "./obj/ex3/Antenna.obj"
        sionObj = sionnaObject("Antenna",obj_file_path, [0,0,0])
        sionObj.set_sionna_material("itu_metal")
        objectsToAdd = [sionObj]
        env2.load_obj_from_file(objectsToAdd, filepath)
        
        env2.store_simulation_info()
        env2.create_communication_link("Smartphone", smartphonePositions["pos2"], "TelTower", [0,10,50])
        env2.calculate_channel_state()
        env2.simulate_digital_communication(1000,10)

        

        env2.display_stats()

        if (isRender):
            try:
                resolution = [1280,720]
                cameraPos = cameraPositions["place2"]["cameraPos"]
                lookAt = cameraPositions["place2"]["lookAt"]
                env2.preview_the_scene(resolution, cameraPos, lookAt, "example3_2.jpg")
                print("!!! RENDER DONE !!!")
            except Exception as e:
                print(e)
    elif(place==3):
        
        """
        POSITION 3
        
        """

        print("---------------------------------- POS AGORA -------------")

        #setup enviroment and start simulation
        scene = load_scene(filepath)
        env3 = SionnaEnv(scene, frequency, bandwith, fft_size,  False, 6, 4, False, VERBOSE=False)

        #add the antenna
        obj_file_path = "./obj/ex3/Antenna.obj"
        sionObj = sionnaObject("Antenna",obj_file_path, [0,0,0])
        sionObj.set_sionna_material("itu_metal")
        objectsToAdd = [sionObj]
        env3.load_obj_from_file(objectsToAdd, filepath)
        
        env3.store_simulation_info()
        env3.create_communication_link("Smartphone", smartphonePositions["pos3"], "TelTower", [0,10,50])
        env3.calculate_channel_state()
        env3.simulate_digital_communication(1000,10)

        

        env3.display_stats()

        if (isRender):
            try:
                resolution = [1280,720]
                cameraPos = cameraPositions["place3"]["cameraPos"]
                lookAt = cameraPositions["place3"]["lookAt"]
                env3.preview_the_scene(resolution, cameraPos, lookAt, "example3_3.jpg")
                print("!!! RENDER DONE !!!")
            except Exception as e:
                print(e)
                
        


"""
For generating datasets 
"""

def generate_dataset_example1():
    print("--- STARTING DATASET GENERATION FOR EXAMPLE 1 ---")
    
    # 1. load scene 
    filepath = "./scenes/ex1/simple_room.xml"
    scene = load_scene(filepath)
    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64
    
    # Setup environment
    env = SionnaEnv(scene, frequency, bandwith, fft_size, False, 6, 4, False, VERBOSE=False)
    
    # 2. add objects 
    materialsToCheck = {
        "mirror": {"material_name": "mirror", "material_relative_permittivity": 1.0, "material_conductivity": 3.8e7, "material_scattering_coefficient": 0.0, "material_xpd_coefficient": 0.0, "material_scattering_pattern": None, "material_frequency_update_callback": None},
        "mercury_wall": {"material_name": "mercury_wall", "material_relative_permittivity": 1.0, "material_conductivity": 1e6, "material_scattering_coefficient": 0.0, "material_xpd_coefficient": 0.0, "material_scattering_pattern": None, "material_frequency_update_callback": None},
        "elevator": {"material_name": "elevator", "material_relative_permittivity": 1.0, "material_conductivity": 1.4e6, "material_scattering_coefficient": 0.05, "material_xpd_coefficient": 0.1, "material_scattering_pattern": None, "material_frequency_update_callback": None},
        "drywall": {"material_name": "drywall", "material_relative_permittivity": 2.5, "material_conductivity": 0.01, "material_scattering_coefficient": 0.05, "material_xpd_coefficient": 0.1, "material_scattering_pattern": None, "material_frequency_update_callback": None},
    }

    objectsToAdd = []
    
    # Wall 1
    sionObj = sionnaObject("barrier-wall", "./obj/ex1/barrier_wall.stl", [3, 2.00469, 0.422312])
    sionObj.create_custom_material(materialsToCheck["mirror"]["material_name"], materialsToCheck["mirror"]["material_relative_permittivity"], materialsToCheck["mirror"]["material_conductivity"], materialsToCheck["mirror"]["material_scattering_coefficient"], materialsToCheck["mirror"]["material_xpd_coefficient"], materialsToCheck["mirror"]["material_scattering_pattern"], materialsToCheck["mirror"]["material_frequency_update_callback"], setColor=[1.0, 0.0, 0.0])
    objectsToAdd.append(sionObj)

    # Wall 2
    sionObj = sionnaObject("barrier-wall-2", "./obj/ex1/barrier_wall2.stl", [3, 2.00469, 1.90484])
    sionObj.create_custom_material(materialsToCheck["mercury_wall"]["material_name"], materialsToCheck["mercury_wall"]["material_relative_permittivity"], materialsToCheck["mercury_wall"]["material_conductivity"], materialsToCheck["mercury_wall"]["material_scattering_coefficient"], materialsToCheck["mercury_wall"]["material_xpd_coefficient"], materialsToCheck["mercury_wall"]["material_scattering_pattern"], materialsToCheck["mercury_wall"]["material_frequency_update_callback"], setColor=[0.0, 1.0, 0.0])
    objectsToAdd.append(sionObj)

    # Wall 3
    sionObj = sionnaObject("barrier-wall-3", "./obj/ex1/barrier_wall3.stl", [3, 0.936904, 1.2271])
    sionObj.create_custom_material(materialsToCheck["elevator"]["material_name"], materialsToCheck["elevator"]["material_relative_permittivity"], materialsToCheck["elevator"]["material_conductivity"], materialsToCheck["elevator"]["material_scattering_coefficient"], materialsToCheck["elevator"]["material_xpd_coefficient"], materialsToCheck["elevator"]["material_scattering_pattern"], materialsToCheck["elevator"]["material_frequency_update_callback"], setColor=[0.0, 0.0, 1.0])
    objectsToAdd.append(sionObj)

    # Wall 4
    sionObj = sionnaObject("barrier-wall-4", "./obj/ex1/barrier_wall4.stl", [3, 3.34588, 1.2271])
    sionObj.create_custom_material(materialsToCheck["drywall"]["material_name"], materialsToCheck["drywall"]["material_relative_permittivity"], materialsToCheck["drywall"]["material_conductivity"], materialsToCheck["drywall"]["material_scattering_coefficient"], materialsToCheck["drywall"]["material_xpd_coefficient"], materialsToCheck["drywall"]["material_scattering_pattern"], materialsToCheck["drywall"]["material_frequency_update_callback"], setColor=[1.0, 1.0, 0.0])
    objectsToAdd.append(sionObj)

    # load objects 
    env.load_obj_from_file(objectsToAdd, filepath)
    
    # simulation
    env.store_simulation_info()

    # 3. LOOP DATASET
<<<<<<< Updated upstream
    # We are moving the receiver to y axis
    # we are gonna get 20 samples
    y_positions = np.linspace(-3.0, 3.0, 20) 
    
    for i, y_pos in enumerate(y_positions):
        
        
        tx_pos = [1.5, -2, 1.5] 
        rx_pos = [4.5, y_pos, 1.5]
=======
    y_positions = np.linspace(0.5, 3.5, 30) 
    
    for i, y_pos in enumerate(y_positions):
        tx_pos = [1.5, 2.0, 1.5] 
        rx_pos = [4.5, float(y_pos), 1.5]
>>>>>>> Stashed changes
        
        if "Router" in env.scene.receivers:
            env.scene.remove("Router")
        if "Laptop" in env.scene.transmitters:
            env.scene.remove("Laptop")
            
        env.create_communication_link("Laptop", tx_pos, "Router", rx_pos, rx_orientation=[0,0,3.14])
        
        try:
            env.calculate_channel_state()
            env.save_dataset_pair(pair_id=i, folder="dataset_ex1")
        except SystemExit:
            print(f"--> SKIPPING Point {i} (y={y_pos:.2f}): No signal found.")
            continue
        except Exception as e:
            print(f"--> ERROR at Point {i}: {e}")
            continue

    print("--- DATASET GENERATION COMPLETED FOR EXAMPLE 1 ---")

def generate_dataset_example2():
    print("--- STARTING DATASET GENERATION FOR EXAMPLE 2 ---")
    
    filepath = "./scenes/ex2/Futuristic_Apartment.xml"
    scene = load_scene(filepath)
    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64
    
    env = SionnaEnv(scene, frequency, bandwith, fft_size, False, 6, 4, False, VERBOSE=False)
    
    materialsToCheck = {
        "mirror": {"material_name": "mirror", "material_relative_permittivity": 1.0, "material_conductivity": 3.8e7, "material_scattering_coefficient": 0.0, "material_xpd_coefficient": 0.0, "material_scattering_pattern": None, "material_frequency_update_callback": None}
    }

    obj_file_path = "./obj/ex2/Mirror.obj"
    sionObj = sionnaObject("Mirror", obj_file_path, [-1, -7.6, 3])
    sionObj.create_custom_material(materialsToCheck["mirror"]["material_name"], materialsToCheck["mirror"]["material_relative_permittivity"], materialsToCheck["mirror"]["material_conductivity"], materialsToCheck["mirror"]["material_scattering_coefficient"], materialsToCheck["mirror"]["material_xpd_coefficient"], materialsToCheck["mirror"]["material_scattering_pattern"], materialsToCheck["mirror"]["material_frequency_update_callback"], setColor=[0.0, 0.0, 1.0])
    
    env.load_obj_from_file([sionObj], filepath)
    env.store_simulation_info()

    # Move Router along Y axis in the apartment
    y_positions = np.linspace(-8.0, -2.0, 30)
    
    for i, y_pos in enumerate(y_positions):
        tx_pos = [6.0, 4.0, 3.0]
        rx_pos = [5.0, float(y_pos), 3.0]
        
        if "Router" in env.scene.receivers:
            env.scene.remove("Router")
        if "Laptop" in env.scene.transmitters:
            env.scene.remove("Laptop")
            
        # Router faces towards the transmitter area
        env.create_communication_link("Laptop", tx_pos, "Router", rx_pos, rx_orientation=[0, 0, 1.57])
        
        try:
            env.calculate_channel_state()
            env.save_dataset_pair(pair_id=i, folder="dataset_ex2")
        except SystemExit:
            print(f"--> SKIPPING Point {i}: No signal found.")
            continue
        except Exception as e:
            print(f"--> ERROR at Point {i}: {e}")
            continue

    print("--- DATASET GENERATION COMPLETED FOR EXAMPLE 2 ---")

def generate_dataset_example3():
    print("--- STARTING DATASET GENERATION FOR EXAMPLE 3 ---")
    
    filepath = "./scenes/ex3/Ioannina.xml"
    scene = load_scene(filepath)
    frequency = 2.437e9
    bandwith = 20e6
    fft_size = 64
    
    env = SionnaEnv(scene, frequency, bandwith, fft_size, False, 6, 4, False, VERBOSE=False)
    
    obj_file_path = "./obj/ex3/Antenna.obj"
    sionObj = sionnaObject("Antenna", obj_file_path, [0, 0, 0])
    sionObj.set_sionna_material("itu_metal")
    
    env.load_obj_from_file([sionObj], filepath)
    env.store_simulation_info()

    smartphonePositions = {
        "pos1": np.array([-8, -31.5, 3.6]),
        "pos2": np.array([-52.1, -237.1, 3.6]),
        "pos3": np.array([-141.3, -115.1, 5.5])
    }

    # Interpolate path
    path1 = np.linspace(smartphonePositions["pos1"], smartphonePositions["pos2"], 15)
    path2 = np.linspace(smartphonePositions["pos2"], smartphonePositions["pos3"], 15)
    all_positions = np.concatenate([path1, path2], axis=0)
    
    for i, pos in enumerate(all_positions):
        tx_pos = [0.0, 10.0, 50.0] # TelTower
        rx_pos = pos.tolist()
        
        if "TelTower" in env.scene.transmitters:
            env.scene.remove("TelTower")
        if "Smartphone" in env.scene.receivers:
            env.scene.remove("Smartphone")
            
        # RX (Smartphone) looks towards the tower
        direction = np.array(tx_pos) - np.array(rx_pos)
        angle = np.arctan2(direction[1], direction[0])
        env.create_communication_link("TelTower", tx_pos, "Smartphone", rx_pos, rx_orientation=[0, 0, float(angle)])
        
        try:
            env.calculate_channel_state()
            env.save_dataset_pair(pair_id=i, folder="dataset_ex3")
        except SystemExit:
            print(f"--> SKIPPING Point {i}: No signal found.")
            continue
        except Exception as e:
            print(f"--> ERROR at Point {i}: {e}")
            continue

    print("--- DATASET GENERATION COMPLETED FOR EXAMPLE 3 ---")

    
#Main
if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path for input <file>.json")
    parser.add_argument("--render", action="store_true", default=False, help="Render task")
    parser.add_argument("--example1", action="store_true", default=False, help="Run example1")
    parser.add_argument("--example2", action="store_true", default=False, help="Run example2")
    parser.add_argument("--example3_1", action="store_true", default=False, help="Run example3_1")
    parser.add_argument("--example3_2", action="store_true", default=False, help="Run example3_2")
    parser.add_argument("--example3_3", action="store_true", default=False, help="Run example3_3")
    parser.add_argument("--gen_dataset", type=int, default=0, help="Generate Dataset from Example (1, 2, or 3)")
    
    args = parser.parse_args()

    if args.gen_dataset == 1:
        print("--- STARTING DATASET GENERATION FOR EXAMPLE 1 ---")
        generate_dataset_example1()
        exit()
    elif args.gen_dataset == 2:
        print("--- STARTING DATASET GENERATION FOR EXAMPLE 2 ---")
        generate_dataset_example2()
        exit()
    elif args.gen_dataset == 3:
        print("--- STARTING DATASET GENERATION FOR EXAMPLE 3 ---")
        generate_dataset_example3()
        exit()
    
    isRunExample1 = args.example1
    isRunExample2 = args.example2
    isRunExample3_1 = args.example3_1
    isRunExample3_2 = args.example3_2
    isRunExample3_3 = args.example3_3
    isRender = args.render
    print("Render: ", args.render)

    if (isRunExample1):
        example1(isRender)
        exit()
    elif (isRunExample2):
        example2(isRender)
        exit()
    elif (isRunExample3_1):
        example3(isRender, 1)
        exit()
    elif (isRunExample3_2):
        example3(isRender, 2)
        exit()
    elif (isRunExample3_3):
        example3(isRender, 3)
        exit()
        
    print("Config: ", args.config)
    filepath = args.config
    
    if (len(filepath)==0):
        print("Parameter is missing:  --config <file>.json")
        exit()

    f = open(filepath, "r")
    example_config = json.load(f)

    filepath = example_config["scene_file"]
    scene = load_scene(filepath)
    frequency = example_config["frequency"]
    bandwith = example_config["bandwith"]
    fft_size = example_config["fft_size"]
    
    rt_calc_diffraction = example_config["rt_calc_diffraction"]
    rt_max_depth = example_config["rt_max_depth"]
    rt_max_parallel_links = example_config["rt_max_parallel_links"]
    est_csi = example_config["est_csi"]
    verbose = example_config["verbose"]

    env = SionnaEnv(scene, frequency, bandwith, fft_size,  rt_calc_diffraction, rt_max_depth, rt_max_parallel_links, est_csi, VERBOSE=verbose)

    sionnObjects = [] 
    if ("objects" in list(example_config.keys())):
        for sionObj in list(example_config["objects"].keys()):
            obj_file_path = example_config["objects"][sionObj]["obj_file_path"]
            put_sionObj = sionnaObject(example_config["objects"][sionObj]["name"], obj_file_path, example_config["objects"][sionObj]["position"])
            if ("custom_material" in list(example_config["objects"][sionObj].keys())):
                update_material_scattering_pattern = None
                if (example_config["objects"][sionObj]["custom_material"]["material_scattering_pattern"] != "None"):
                    update_material_scattering_pattern = example_config["objects"][sionObj]["custom_material"]["material_scattering_pattern"]
                update_material_frequency_update_callback = None
                if (example_config["objects"][sionObj]["custom_material"]["material_frequency_update_callback"] != "None"):
                    update_material_frequency_update_callback = example_config["objects"][sionObj]["custom_material"]["material_frequency_update_callback"]

                put_sionObj.create_custom_material(
                    material_name=example_config["objects"][sionObj]["custom_material"]["material_name"],
                    material_relative_permittivity=example_config["objects"][sionObj]["custom_material"]["material_relative_permittivity"],
                    material_conductivity=example_config["objects"][sionObj]["custom_material"]["material_conductivity"],
                    material_scattering_coefficient=example_config["objects"][sionObj]["custom_material"]["material_scattering_coefficient"],
                    material_xpd_coefficient=example_config["objects"][sionObj]["custom_material"]["material_xpd_coefficient"],
                    material_scattering_pattern=update_material_scattering_pattern,
                    material_frequency_update_callback=update_material_frequency_update_callback,
                    setColor=example_config["objects"][sionObj]["custom_material"]["set_color"]
                )
            elif ("material" in list(example_config["objects"][sionObj].keys())):
                put_sionObj.set_sionna_material(example_config["objects"][sionObj]["material"])
            else:
                print("No material has been declared in your new items")
                exit()
            sionnObjects.append(put_sionObj)
        env.load_obj_from_file(sionnObjects, filepath)

    env.store_simulation_info()
    env.create_communication_link(example_config["tx_name"], example_config["tx_position"], example_config["rx_name"], example_config["rx_position"])
    env.calculate_channel_state()
    env.simulate_digital_communication(example_config["batch_size"],example_config["iter"])
    env.display_stats()
    
    if (isRender):
        try:
            resolution = example_config["resolution"]
            cameraPos = example_config["camera_positions"]
            lookAt = example_config["look_at"]
            simulation_name = example_config["simulation_name"]
            render_file_output = example_config["render_output"]
            env.preview_the_scene(resolution, cameraPos, lookAt, render_file_output)
            print("RENDER DONE")
        except Exception as e:
            print(e)



    