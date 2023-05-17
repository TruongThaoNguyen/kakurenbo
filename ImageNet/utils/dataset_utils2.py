import torch

import numpy as np
import os
import os.path
import re
import sys

import time
import zipfile
import horovod.torch as hvd


def download_dataset_to_ssd(rank_per_node,folders_path, out):
    start = time.time()
    root = os.path.abspath(folders_path)
    local_folder = os.path.abspath(out)
    
    rank = hvd.rank()
    size = hvd.size()
    rank_per_node = int(rank_per_node)
    local_rank = rank % rank_per_node
    global_rank = rank // rank_per_node
    
    # Create subfolder for each node
    dest_file_name = os.path.join(local_folder, "class_to_idx.txt")
    
    if local_rank ==0:
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
            
    # Reading source folder to count number of partitions
    all_zip_files = []
    for filename in sorted_alphanumeric(os.listdir(root)): #.sort():
        if is_zip_file(filename):
            all_zip_files.append(filename)
    num_partitions = len(all_zip_files)
    # if rank == 0:
        # print(num_partitions,all_zip_files)
        
    zip_files = []
    for i in range(0,num_partitions):
        if i % rank_per_node == local_rank:
            zip_files.append(all_zip_files[i])
            #print("add", i,all_zip_files[i])
        # else:
            # print("dont add", i)
    #print("Rank", rank, "download", len(zip_files), "partitions:", zip_files)
    
    # Copy partitions from root to local
    if local_rank ==0:
        source_file_name = os.path.join(root, "class_to_idx.txt")
        command = "cp " + str(source_file_name) + " " + str(dest_file_name) 
        #print(command)
        os.system(command)

    #print(rank, global_rank, local_rank, rank_per_node)
    
    for i in range(0, len(zip_files)):
        try:
            zip_file_name = zip_files[i]
            source_file_name = os.path.join(root, zip_file_name)
            local_file_name = os.path.join(local_folder, zip_file_name)
            command = "cp " + str(source_file_name) + " " + str(local_folder) 
            #print(rank, command)
            os.system(command)    
            with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
                #command = "ls -la " + local_file_name
                #print(rank, "Extract", command)
                #os.system(command)
                try:
                    zip_ref.extractall(local_folder)
                except FileExistsError as e:
                    try:
                        print(rank, "Error: Re-extract the file", local_file_name, e)
                        zip_ref.extractall(local_folder)
                    except FileExistsError as e2:
                        print(rank, "Error: Re-extract the file 2", local_file_name, e)
                        zip_ref.extractall(local_folder)
            command = "rm " + str(local_file_name)
            #print(command)
            os.system(command)
        except:
            print(rank,"Unexpected error:",zip_files[i])
            continue
    stop = time.time()
    #if local_rank == 0:
    print("Rank", rank,"Finished in {:.10f} second".format(stop - start))
    # print("rank",rank,local_folder)
    # os.system("ls " + str(local_folder))
    return local_folder


def download_dataset_to_ssd_bak(rank_per_node,folders_path, out):
    start = time.time()
    root = os.path.abspath(folders_path)
    local_folder = os.path.abspath(out)
    
    rank = hvd.rank()
    size = hvd.size()
    rank_per_node = int(rank_per_node)
    local_rank = rank % rank_per_node
    global_rank = rank // rank_per_node
    
    # Create subfolder for each node
    dest_file_name = os.path.join(local_folder, "class_to_idx.txt")
    
    if local_rank ==0:
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
            
        # Reading source folder to count number of partitions
        all_zip_files = []
        for filename in sorted_alphanumeric(os.listdir(root)): #.sort():
            if is_zip_file(filename):
                all_zip_files.append(filename)
        num_partitions = len(all_zip_files)
        # if rank == 0:
            # print(num_partitions,all_zip_files)
            
        zip_files = []
        for i in range(0,num_partitions):
                zip_files.append(all_zip_files[i])
                #print("add", i,all_zip_files[i])
            # else:
                # print("dont add", i)
        #print("Rank", rank, "download", len(zip_files), "partitions:", zip_files)
        
        # Copy partitions from root to local
        source_file_name = os.path.join(root, "class_to_idx.txt")
        
        command = "cp " + str(source_file_name) + " " + str(dest_file_name) 
        #print(command)
        os.system(command)

        #print(rank, global_rank, local_rank, rank_per_node)
        
        for i in range(0, len(zip_files)):
            zip_file_name = zip_files[i]
            source_file_name = os.path.join(root, zip_file_name)
            local_file_name = os.path.join(local_folder, zip_file_name)
            command = "cp " + str(source_file_name) + " " + str(local_folder) 
            #print(rank, command)
            os.system(command)
            
            with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
                #command = "ls -la " + local_file_name
                #print(rank, "Extract", command)
                #os.system(command)
                zip_ref.extractall(local_folder)
                
            command = "rm " + str(local_file_name)
            #print(command)
            os.system(command)

        stop = time.time()
        if local_rank == 0:
            print("Rank", rank,"Finished in {:.10f} second".format(stop - start))
    # print("rank",rank,local_folder)
    # os.system("ls " + str(local_folder))
    return local_folder

def download_dataset(folders_path, out):
    start = time.time()
    root = os.path.abspath(folders_path)
    out_folder = os.path.abspath(out)
    
    rank = hvd.rank()
    size = hvd.size()
    
    # Reading source folder to count number of partitions
    all_zip_files = []
    for filename in sorted_alphanumeric(os.listdir(root)): #.sort():
        if is_zip_file(filename):
            all_zip_files.append(filename)
    num_partitions = len(all_zip_files)
    if rank == 0:
        print(num_partitions,all_zip_files)
    zip_files = []
    for i in range(0,num_partitions):
        if i % size == rank:
            zip_files.append(all_zip_files[i])
            #print("add", i,all_zip_files[i])
        # else:
            # print("dont add", i)
    print("Rank", rank, "download", len(zip_files), "partitions:", zip_files)
    
    # Create subfolder for each rank
    local_folder = os.path.join(out_folder,str(rank))
    command = "rm -r " + str(local_folder)
    os.system(command)
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    
    # Copy partitions from root to local
    source_file_name = os.path.join(root, "class_to_idx.txt")
    dest_file_name = os.path.join(local_folder, "class_to_idx.txt")
    command = "cp " + str(source_file_name) + " " + str(dest_file_name) 
    #print(command)
    os.system(command)
    
    for i in range(0, len(zip_files)):
        zip_file_name = zip_files[i]
        source_file_name = os.path.join(root, zip_file_name)
        local_file_name = os.path.join(local_folder, zip_file_name)
        command = "cp " + str(source_file_name) + " " + str(local_folder) 
        #print(command)
        os.system(command)
        
        with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
            try:
                zip_ref.extractall(local_folder)
            except FileExistsError as e:
                print(rank, "Error: Re-extract the file", e)
                zip_ref.extractall(local_folder)
            
        command = "rm " + str(local_file_name)
        #print(command)
        os.system(command)
        
    stop = time.time()
    print("Rank", rank,"Finished in {:.10f} second".format(stop - start))
    #os.system("ls " + str(local_folder),)
    return local_folder, dest_file_name

ZIP_EXTENSIONS = [".zip"]

def is_zip_file(filename):
    return any(filename.endswith(extension) for extension in ZIP_EXTENSIONS)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
    