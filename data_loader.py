import re
import numpy as np
import collections
import os

def get_data(end=100, data_dir=""):
    # TODO target loader
    # source_path = os.path.join(data_dir, "source.txt")
    # target_path = os.path.join(data_dir, "label.txt")
    # print(source_path, target_path)
    with open('data/cooper_source.txt','r') as s:#, open('data/cooper_target.txt','r') as t:
        source = ''
        target = ''
        i = 0
        while i < 100:
            source_line = s.readline()
            # target_line = t.readline()
            if not source_line: #or not target_line:
                break
            source_line = re.sub(' +',' ',source_line)
            source_line = re.sub('\n',' ',source_line)
            # target_line = re.sub(' +',' ',target_line)
            # target_line = re.sub('\n',' ',target_line)
            source += source_line
            # target += target_line
            i+=1
        source = list(source)
        # target = list(target)
        print(len(source))
        # print(len(target))
        # if len(source) != len(target):
            # raise ValueError("The length of source and target does not match")
        for i in range(len(source)):
            source[i] = ord(source[i])
            # target[i] = ord(target[i])
        source = np.asarray(source).astype(np.float32,copy=False)
        # target = np.asarray(target).astype(np.float32,copy=False)
        return (source, target)


