import os
# os.environ['LD_LIBRARY_PATH'] = '/usr/lib/llvm-13/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
# os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/llvm-13/lib/libLLVM.so'
# os.environ['MI_DEFAULT_VARIANT'] = 'llvm_ad_rgb'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# print("-----------------HERE PATHS -----------------")
# print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
# print("DRJIT_LIBLLVM_PATH:", os.environ.get('DRJIT_LIBLLVM_PATH'))

import argparse



# Sionna
import os




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
        a_np = np.abs(self.a.numpy().flatten()) # |ak|
        tau_np = self.tau.numpy().flatten() # |tk|
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
                
                
                
                
            
           

            
        
            

                

            

        



        




    
    def display_stats(self):
        
        mat.use("Qt5Agg")
        

        plt.figure(figsize=(15,10))
        plt.title("Channel Informations")
        
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
        display_delays = self.tau.numpy().flatten() #convert to nanoseconds
        plt.stem(display_delays, display_amplitudes, linefmt="b-", markerfmt="bo", basefmt=' ')
        plt.title("Impulse Response (Power)")
        plt.xlabel("Delay [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)

        if (len(display_delays)>0):#case small values
            max_delay = np.max(display_delays) #take the max limit 
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

        #display channel info 
        plt.subplot(2,2,4)
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

        #display PAM BER 
        plt.subplot(2,4,1)
        ber_display = self.modulations["pam"]["ber"]
        plt.semilogy(self.snr_values, ber_display, "-o")
        plt.title("PAM modulation BER")
        plt.xlabel("Eb/No (dB)")
        plt.ylabel("BER")
        plt.grid(True)

        # #fix the limits 
        # plt.ylim(bottom=0.4, top=0.6)
        # plt.xlim(left=min(self.snr_values)-1, right=max(self.snr_values)+1)

        #display PAM BLER
        plt.subplot(2,4,2)
        ber_display = self.modulations["pam"]["bler"]
        plt.semilogy(self.snr_values, ber_display, "-o")
        plt.title("PAM modulation BLER")
        plt.xlabel("Eb/No (dB)")
        plt.ylabel("BLER")
        plt.grid(True)

        #display PAM 



        #display 4-PAM BER 
        plt.subplot(2,4,3)
        ber_display = self.modulations["4-pam"]["ber"]
        plt.semilogy(self.snr_values, ber_display, "-o")
        plt.title("4-PAM modulation BER")
        plt.xlabel("Eb/No (dB)")
        plt.ylabel("BER")
        plt.grid(True)

        #fix the limits 
        # plt.ylim(bottom=0.4, top=0.6)
        # plt.xlim(left=min(self.snr_values)-1, right=max(self.snr_values)+1)

        #display 4-PAM BLER
        plt.subplot(2,4,4)
        ber_display = self.modulations["4-pam"]["bler"]
        plt.semilogy(self.snr_values, ber_display, "-o")
        plt.title("4-PAM modulation BLER")
        plt.xlabel("Eb/No (dB)")
        plt.ylabel("BLER")
        plt.grid(True)



        #display 16-QAM BER 
        plt.subplot(2,4,5)
        ber_display = self.modulations["16-qam"]["ber"]
        plt.semilogy(self.snr_values, ber_display, "-o")
        plt.title("16-QAM modulation BER")
        plt.xlabel("Eb/No (dB)")
        plt.ylabel("BER")
        plt.grid(True)

        #fix the limits 
        # plt.ylim(bottom=0.4, top=0.6)
        # plt.xlim(left=min(self.snr_values)-1, right=max(self.snr_values)+1)

        #display 16-QAM BLER
        plt.subplot(2,4,6)
        ber_display = self.modulations["16-qam"]["bler"]
        plt.semilogy(self.snr_values, ber_display, "-o")
        plt.title("16-QAM modulation BLER")
        plt.xlabel("Eb/No (dB)")
        plt.ylabel("BLER")
        plt.grid(True)




        #display 64-QAM BER 
        plt.subplot(2,4,7)
        ber_display = self.modulations["64-qam"]["ber"]
        plt.semilogy(self.snr_values, ber_display, "-o")
        plt.title("64-QAM modulation BER")
        plt.xlabel("Eb/No (dB)")
        plt.ylabel("BER")
        plt.grid(True)

        #fix the limits 
        # plt.ylim(bottom=0.4, top=0.6)
        # plt.xlim(left=min(self.snr_values)-1, right=max(self.snr_values)+1)

        #display 64-QAM BLER
        plt.subplot(2,4,8)
        ber_display = self.modulations["64-qam"]["bler"]
        plt.semilogy(self.snr_values, ber_display, "-o")
        plt.title("64-QAM modulation BLER")
        plt.xlabel("Eb/No (dB)")
        plt.ylabel("BLER")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        


        # #plot ser
        # plt.figure(figsize=(15,10))
        # plt.title("Symbol Error Rate")
        # pos = 1
        # for mod in self.modulations.keys():
        #     plt.subplot(2,4,pos)
        #     display_values = self.modulations[mod]["ser"]
        #     plt.semilogy(self.snr_values, display_values, "-o")
        #     plt.title(f"{mod} modulation SER")
        #     plt.xlabel("Eb/No (dB)")
        #     plt.ylabel("SER")
        #     plt.grid(True)
        #     pos+=1
        # plt.tight_layout()
        # plt.show()


        #plot Bits Error 
        plt.figure(figsize=(15,10))
        plt.title("Bits Error")
        pos = 1
        for mod in self.modulations.keys():
            plt.subplot(2,4,pos)
            display_values = self.modulations[mod]["bit_errors"]
            plt.semilogy(self.snr_values, display_values, "-o")
            plt.title(f"{mod} modulation Bits Error")
            plt.xlabel("Eb/No (dB)")
            plt.ylabel("Bits Error")
            plt.grid(True)
            pos+=1
        plt.tight_layout()
        plt.show()

        
        # #plot Block Error 
        # plt.figure(figsize=(15,10))
        # plt.title("Block Error")
        # pos = 1
        # for mod in self.modulations.keys():
        #     plt.subplot(2,4,pos)
        #     display_values = self.modulations[mod]["block_errors"]
        #     plt.semilogy(self.snr_values, display_values, "-o")
        #     plt.title(f"{mod} modulation Block Error")
        #     plt.xlabel("Eb/No (dB)")
        #     plt.ylabel("Block Error")
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
    env.simulate_digital_communication(1000,10)

    

    env.display_stats()

    if (isRender):
        #render 
        try:
            env.preview_the_scene([1280,720],[-1.5,3,6],[4.5,2,1], "example1.jpg")
            print("!!! RENDER DONE !!!")
        except Exception as e:
            print(e)
    
    

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

 

   
        
    

def example3(isRender):

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
            
    

     

    
# #Test examples 
# if __name__ == '__main__': 
    

    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--single_run", help="Whether not to terminate after single run", action='store_true')
#     parser.add_argument("--rt_calc_diffraction", help="Calc diffraction in raytracing", action='store_true')
#     parser.add_argument("--rt_max_depth", type=int, default=6, help="Calc diffraction in raytracing")
#     parser.add_argument("--rt_max_parallel_links", type=int, default=4, help="Max no. of receivers")
#     parser.add_argument("--est_csi", help="Whether to estimate complex CSI per OFDM subcarrier", action='store_true')
#     parser.add_argument("--verbose", help="Whether to run in verbose mode", action='store_true')
    
    
#     args = parser.parse_args()

#     #test
#     print("Single_run: ", args.single_run)
#     print("rt_calc_diffraction: ", args.rt_calc_diffraction)
#     print("rt_max_depth: ", args.rt_max_depth)
#     print("rt_max_parallel_links: ", args.rt_max_parallel_links)
#     print("est_csi: ", args.est_csi)
#     print("verbose: ", args.verbose)
#     #end test


#     #print("HELLO SIONNA!!!")

#     example1()

#     #example2() 

#     #example3()   

    
#Main
if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path for input <file>.json")
    parser.add_argument("--render", action="store_true", default=False, help="Render task")
    parser.add_argument("--example1", action="store_true", default=False, help="Run example1")
    parser.add_argument("--example2", action="store_true", default=False, help="Run example2")
    parser.add_argument("--example3", action="store_true", default=False, help="Run example3")
    args = parser.parse_args()
    
    isRunExample1 = args.example1
    isRunExample2 = args.example2
    isRunExample3 = args.example3
    isRender = args.render
    print("Render: ", args.render)

    if (isRunExample1):
        example1(isRender)
        exit()
    elif (isRunExample2):
        example2(isRender)
        exit()
    elif (isRunExample3):
        example3(isRender)
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
    

    #setup enviroment and start simulation
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

            put_sionObj = sionnaObject(example_config["objects"][sionObj]["name"],
                                    obj_file_path, 
                                    example_config["objects"][sionObj]["position"]
                                    )
            #print(list(example_config["objects"][sionObj].keys()))
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

    


    