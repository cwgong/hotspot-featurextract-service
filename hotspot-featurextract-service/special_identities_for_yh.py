# -*- coding: utf-8 -*-

import codecs
import os

def visit_dir_or_file(input_dir, file_paths):
    
    if os.path.isdir(input_dir):
        files = os.listdir(input_dir)
        for file in files:
            visit_dir_or_file(input_dir+"/"+file, file_paths)
    else:

        if os.path.isfile(input_dir) and input_dir.endswith(".txt"): 
            file_paths.append(input_dir) 
            
def read_file(filepath):
    
    file = codecs.open(filepath, 'r', encoding = 'utf-8')
    lines = file.readlines()
    
    return lines
    
def write_to_file(string, file_path, param):

    out = codecs.open(file_path, param, 'utf-8')
    out.write(string)
    out.close


special_identities = []
lines = read_file('./securities_name.txt')
for line in lines:
    special_identities.append(line.strip())
    
SPECIAL_IDENTITIES = ['证券部人士', '投行人士', '投行业内人士', '创投界人士', '市场分析人士','私募分析人士','专家','资深专家','从业人士','资深专家及从业人士']
for x in SPECIAL_IDENTITIES:
    special_identities.append(x)
    
ssqy = ''
jj = ''

for rw in special_identities:
    
    l = ('''INSERT INTO `specialist` (`person_type`, `person_name`, `company`, `introduction`, `create_at`, `update_at`) VALUES (0, \'%s\', \'%s\', \'%s\', UNIX_TIMESTAMP()*1000, UNIX_TIMESTAMP()*1000);''' % (rw, ssqy, jj))
    write_to_file(l +'\n', './name_for_sql.txt', 'a')
    