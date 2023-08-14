import pickle
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import os
from sklearn.metrics import classification_report

import random
import math
from typing import List, Tuple, Dict
from pathlib import Path
from copy import deepcopy
import hashlib
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
from stego import FloatBinary, str_to_bits, bits_to_str, dummy_data_generator
import csv
import json


def hash_str(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

import csv

def csv_dict(csv_solution):
    dict_ans = {}
    with open(csv_solution, newline='') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            dict_ans[', '.join(row).split(',')[0]] = ', '.join(row).split(',')[1]
    return dict_ans

def class_json_loader(json_location):
    data = {}
    # Opening JSON file
    with open(json_location) as json_file:
        data = json.load(json_file)
    
    return data

def generate_non_repeating_list(length, start, end):
    if length > (end - start + 1):
        length = (end - start + 1)
        #raise ValueError("Desired length is greater than the available range.")
    
    num_set = set()
    result = []
    
    while len(result) < length:
        # Generate a random number with bias towards lower values
        bias_factor = 2
        num = num = int(np.random.exponential(scale=bias_factor)) + start
        
        if num not in num_set:
            num_set.add(num)
            result.append(num)
    
    return result

class ModelOp:

    BITS_TO_USE = 8
    DATASET_FOLDER = ""
    true_labels = []
    FILES_TO_TEST_ON = []
    class_dict = {}
    model = None
    model_layers_capacity = None
    overall_storage_capacity_bytes=0

    def __init__(self, float_bits_to_use, data_folder,file_type = '.JPEG',not_imagenet_class = False,csv_solution = '/media/sf_shared/pics/LOC_val_solution.csv',json_location = '/media/sf_shared/pics/imagenet_class_index.json'):
        print(file_type)

        #------------------------------------------
        #only float point can be useed to hide data
        #------------------------------------------
        if float_bits_to_use > 23:
            print("the amount need to be 23 or less - default set to 8")
        else:
            self.BITS_TO_USE = float_bits_to_use

        self.DATASET_FOLDER = data_folder
        
        if(not not_imagenet_class):
            self.load_files(self.DATASET_FOLDER, file_type,csv_solution)
            self.class_dict = class_json_loader(json_location)

    def load_files(self, new_data_folder,file_type,csv_solution):
        '''
        for now we use only models thats get a photo as input
        '''
        self.DATASET_FOLDER = new_data_folder
        self.FILES_TO_TEST_ON = list(map(str, Path(self.DATASET_FOLDER).glob(f"**/*{file_type}")))
        assert len(self.FILES_TO_TEST_ON) > 0, "You'll need some images to test the network performance"

        dic = csv_dict(csv_solution)
        for f in self.FILES_TO_TEST_ON:
            filename = f.split('/')[-1]
            self.true_labels.append(dic[filename.split(".")[0]])

    def get_photo_list(self):
        return self.FILES_TO_TEST_ON

    def preprocess_image(self, image_path):
        # Define the transformations to be applied to the input image
        transform = transforms.Compose([
            transforms.Resize((255, 255)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Load the image and apply the defined transformations
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        return image

    def load_model(self, filepath):
        """
        filepath = none means use the deafult model I choose
        """
        model_fp32 = None
        if filepath == "ResNet101":
            model_fp32 = models.resnet101(pretrained=True)
        elif filepath == "Vgg19":
            model_fp32 = models.vgg19(pretrained=True)
        elif filepath == "Vgg16":
            model_fp32 = models.vgg16(pretrained=True)
        elif filepath == "Inception":
            model_fp32 = models.inception_v3(pretrained=True)
        elif filepath == "ResNet50":
            model_fp32 = models.resnet50(pretrained=True)
        elif filepath == "Mobilenet":
            model_fp32 = models.mobilenet_v3_large(pretrained=True)
        elif filepath == "ResNet18":
            model_fp32 = models.resnet18(pretrained=True)
        else:
            with open(filepath,"rb") as f:
                model_fp32 = pickle.load(f)
        model_fp32.eval()   
        self.model = model_fp32
        self.get_model_data_capacity()
    
    def get_model_data_capacity(self):
        """
        Returns the model capacity - model needs to be loaded.
        """
        assert self.model is not None, "No model has been loaded"
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                count += 1

        layers_storage_capacity_mb = {}
        p = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                nb_params = np.prod(module.weight.data.shape)
                capacity_in_bytes = np.floor((nb_params * self.BITS_TO_USE) / 8).astype(int)
                layers_storage_capacity_mb[name] = capacity_in_bytes / float(1<<20) # Convert to MB
                p[name] = nb_params
        print("Sum of MB to hide in Conv2D:", sum(layers_storage_capacity_mb.values()), "Count of Conv2D:", len(layers_storage_capacity_mb.values()))

        self.model_layers_capacity = layers_storage_capacity_mb
        return self.model_layers_capacity

    def get_simple_statistics(self, p=False):
        layer_names = list(self.model_layers_capacity.keys())
        selected_layers_weights = []
        for n in layer_names:
            module = self.model
            for name in n.split('.'):
                module = getattr(module, name)
            v = module.weight.detach().numpy().ravel()
            selected_layers_weights.extend(v)
        selected_layers_weights = np.array(selected_layers_weights)

        nb_values = len(selected_layers_weights)
        self.float_amount = nb_values
        min_value = selected_layers_weights.min()
        abs_min_value = np.abs(selected_layers_weights).min()
        max_value = selected_layers_weights.max()
        mean_value = selected_layers_weights.mean()
        nb_really_small_values = (np.abs(selected_layers_weights) < 10e-4).sum()
        nb_small_values = (np.abs(selected_layers_weights) < 10e-3).sum()
        nb_negative_values = (selected_layers_weights < 0).sum()
        nb_positive_values = (selected_layers_weights > 0).sum()
        self.overall_storage_capacity_bytes = nb_values * self.BITS_TO_USE / 8
        overall_storage_capacity_mb = self.overall_storage_capacity_bytes // float(1 << 20)
        stats = {
            "nb_values": nb_values,
            "min_value": min_value,
            "abs_min_value": abs_min_value,
            "max_value": max_value,
            "mean_value": mean_value,
            "nb_really_small_values": nb_really_small_values,
            "nb_small_values": nb_small_values,
            "nb_negative_values": nb_negative_values,
            "nb_positive_values": nb_positive_values,
            "overall_storage_capacity_bytes": self.overall_storage_capacity_bytes,
            "overall_storage_capacity_mb": overall_storage_capacity_mb
        }
        if p:
            print(f"""Stats for ALL LAYERS
            #         ---
            #         Min: {min_value}
            #         Abs. Min {abs_min_value}
            #         Max: {max_value}
            #         Mean: {mean_value}
            #         ---
            #         Nb total values: {nb_values}
            #         Nb values < 10e-4: {nb_really_small_values} - {nb_really_small_values / nb_values * 100:.4f}%
            #         Nb values < 10e-3: {nb_small_values} - {nb_small_values / nb_values * 100:.4f}%
            #         Nb negatives: {nb_negative_values} - {nb_negative_values / nb_values * 100:.4f}%
            #         Nb positives: {nb_positive_values} - {nb_positive_values / nb_values * 100:.4f}%
            #         ---
            #         (Maximum) Storage capacity is {overall_storage_capacity_mb} MB for the {len(layer_names)} layers with the {self.BITS_TO_USE} bits modification
            #         """)
        return stats


    def insert_file_data(self, path: str,padding):
        data = ''
        with open(path,"rb") as f:
            data = list(f.read())

        layer_names = list(self.model_layers_capacity.keys())
        print(len(data))
        for i in padding:
            data.append(0)
        secret_bits = data
        nb_vals_needed = math.ceil(len(secret_bits) / self.BITS_TO_USE)
        print(f"We need {nb_vals_needed} float values to store the info\nOverall number of values we could use: {self.float_amount}")

        # This dict will hold the modified (secret hidden) weights for the layers
        modified_weights_dict = {}

        # We are modifying the layers in a defined order to know what was changed exactly
        # This order is needed when we would like to recover the message

        # Variable which tracks the number of values changed so far (used to index the secret message bits)
        i = 0

        for n in layer_names:
            # Check if we need more values to use to hide the secret, if not then we are done with modifying the layer's weights
            if i >= nb_vals_needed:
                break

            module = self.model
            for name in n.split('.'):
                module = getattr(module, name)
            weights = module.weight.data

            w_shape = weights.shape
            w = weights.view(-1)

            nb_params_in_layer = w.shape[0]

            for j in range(nb_params_in_layer):
                # Chunk of data from the secret to hide
                _from_index = i * self.BITS_TO_USE
                _to_index = _from_index + self.BITS_TO_USE
                bits_to_hide = secret_bits[_from_index:_to_index]
                bits_to_hide = list(map(bool, bits_to_hide))

                # Modify the defined bits of the float value fraction
                x = FloatBinary(w[j].item())
                fraction_modified = list(x.fraction)
                if len(bits_to_hide) > 0:
                    fraction_modified[-self.BITS_TO_USE:] = bits_to_hide

                x_modified = x.modify_clone(fraction=tuple(fraction_modified))
                w[j] = torch.tensor(x_modified.v)

                i += 1

                # Check if we need more values to use to hide the secret in the current layer, if not then we are done
                if i >= nb_vals_needed:
                    break

            modified_weights_dict[n] = weights.view(w_shape)

    def insert_str_data(self, data: str):
        layer_names = list(self.model_layers_capacity.keys())
        secret_bits = str_to_bits(data)
        nb_vals_needed = math.ceil(len(secret_bits) / self.BITS_TO_USE)
        print(f"We need {nb_vals_needed} float values to store the info\nOverall number of values we could use: {self.float_amount}")

        # This dict will hold the modified (secret hidden) weights for the layers
        modified_weights_dict = {}

        # We are modifying the layers in a defined order to know what was changed exactly
        # This order is needed when we would like to recover the message

        # Variable which tracks the number of values changed so far (used to index the secret message bits)
        i = 0

        for n in layer_names:
            # Check if we need more values to use to hide the secret, if not then we are done with modifying the layer's weights
            if i >= nb_vals_needed:
                break

            module = self.model
            for name in n.split('.'):
                module = getattr(module, name)
            weights = module.weight.data

            w_shape = weights.shape
            w = weights.view(-1)

            nb_params_in_layer = w.shape[0]

            for j in range(nb_params_in_layer):
                # Chunk of data from the secret to hide
                _from_index = i * self.BITS_TO_USE
                _to_index = _from_index + self.BITS_TO_USE
                bits_to_hide = secret_bits[_from_index:_to_index]
                bits_to_hide = list(map(bool, bits_to_hide))

                # Modify the defined bits of the float value fraction
                x = FloatBinary(w[j].item())
                fraction_modified = list(x.fraction)
                if len(bits_to_hide) > 0:
                    fraction_modified[-self.BITS_TO_USE:] = bits_to_hide

                x_modified = x.modify_clone(fraction=tuple(fraction_modified))
                w[j] = torch.tensor(x_modified.v)

                i += 1

                # Check if we need more values to use to hide the secret in the current layer, if not then we are done
                if i >= nb_vals_needed:
                    break

            modified_weights_dict[n] = weights.view(w_shape)

    def recover(self, bits_to_show=100):
        # We store the extracted bits of data here
        hidden_data: List[bool] = []
        layer_names = list(self.model_layers_capacity.keys())

        for n in layer_names:
            module = self.model
            for name in n.split('.'):
                module = getattr(module, name)
            weights = module.weight.data

            w_shape = weights.shape
            w = weights.view(-1)

            nb_params_in_layer = w.shape[0]

            for i in range(nb_params_in_layer):
                x = FloatBinary(w[i].item())
                hidden_bits = x.fraction[-self.BITS_TO_USE:]
                hidden_data.extend(hidden_bits)

        recovered_message: str = bits_to_str(list(map(int, hidden_data[:bits_to_show])))
        return recovered_message

    def get_mtd_bool_list(self, strategy, bits = 1,maxim=1):
        '''
        Create a Moving Target Defense layer based on randomly generated booleans
        '''
        rand_list = []
        if strategy == 'uniform':
            for i in range(bits):
                rand_list.append(bool(torch.randint(0, 2, (1,)).item()))
        elif strategy == 'n_random':
            #print(bits,maxim)
            who_to_change = generate_non_repeating_list(bits,maxim,22)
            for i in range(23):
                if i in who_to_change:
                    rand_list.append(bool(torch.randint(0, 2, (1,)).item()))
                else: 
                    rand_list.append(None)
        return rand_list

    def disarm(self, strategy, bits = 1):
        
        if strategy == 'uniform' or strategy == 'n_random':
            layer_names = list(self.model_layers_capacity.keys())
            for n in layer_names:
                module = self.model
                for name in n.split('.'):
                    module = getattr(module, name)

                weights = module.weight.data
                w_shape = weights.shape
                w = weights.view(-1)

                nb_params_in_layer = w.shape[0]

                for j in range(nb_params_in_layer):
                    x = FloatBinary(w[j].item())
                    fraction_modified = list(x.fraction)
                    if strategy == 'n_random':
                        end = 0
                        for i in range(23):
                            if fraction_modified[i] == False:
                                end += 1
                            else:
                                break 
                        bits_to_replace = self.get_mtd_bool_list(strategy,bits,end)
                        #print(fraction_modified)
                        for indx, varb in enumerate(bits_to_replace):
                            if varb == False or varb == True:
                                fraction_modified[22-indx] = varb
                        #print(fraction_modified,bits_to_replace)
                    else:
                        bits_to_replace = self.get_mtd_bool_list(strategy,bits)
                        fraction_modified[-bits:] = bits_to_replace
                    x_modified = x.modify_clone(fraction=tuple(fraction_modified))
                    w[j] = x_modified.v

        elif strategy == "quanti8":
            model_int8 = torch.ao.quantization.quantize_dynamic(
            self.model,  # the original model  # a set of layers to dynamically quantize
            dtype=torch.qint8)  # the target dtype for quantized weights
            self.model = model_int8
            

    def test_model(self):
        preds = []
        runner = 0
        for f in self.FILES_TO_TEST_ON:
            try:
                runner += 1
                with torch.no_grad():
                    input_image = self.preprocess_image(f)
                    output = self.model(input_image)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                class_idx = torch.argmax(probabilities).item()
                idx = self.class_dict[str(class_idx)]
                preds.append(idx[0])
                print(f'{runner}/50000 Predicted class: {idx[1]}, Probability: {probabilities[class_idx].item()}')
            except:
                preds.append(1002)
                print(f'{runner}/50000 Predicted class: 1002, Probability: CHANNELS ERROR')
        prec = classification_report(preds, self.true_labels, output_dict=True)['accuracy']
        print(prec*100,"%")
        return prec