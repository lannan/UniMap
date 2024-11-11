import time
import os
import pickle
import re
import json
from collections import defaultdict

x86_reg_re = ("rip,?|rax,?|rbx,?|rcx,?|rdx,?|rsp,?|rbp,?|rsi,?|rdi,?|eax,?|ecx,?|edx,?|ebx,?|"
             "esp,?|ebp,?|esi,?|edi,?|ax,?|cx,?|dx,?|bx,?|sp,?|bp,?|di,?|si,?|"
             "ah,?|al,?|ch,?|cl,?|dh,?|dl,?|bh,?|bl,?|spl,?|bpl,?|sil,?|dil,?|")

# ref https://blog.yossarian.net/2020/11/30/How-many-registers-does-an-x86-64-cpu-have
bounds_regi = set(['bnd0', 'bnd0,', 'bnd1', 'bnd1,', 'bnd2', 'bnd2,', 'bnd3', 'bnd3,', 'bndcfg', 'bndcfg,', 'bndcfu', 'bndcfu,', 'bndstatus', 'bndstatus,'])
debug_regi = set(['dr' + str(i) for i in range(8)] + ['dr' + str(i) + ',' for i in range(8)])
control_regi = set(['cr' + str(i) for i in range(16)] + ['cr,' + str(i) for i in range(16)])
stack_regi = set(['st(' + str(i) + ')' for i in range(8)] + ['st(' + str(i) + '),' for i in range(8)])
sse_regi = set(['xmm' + str(i)  for i in range(32)]  + ['xmm' + str(i) + ','  for i in range(32)])
avx_regi = set(['zmm' + str(i)  for i in range(32)] + ['zmm' + str(i) + ','  for i in range(32)]) 
av2_regi = set(['ymm' + str(i)  for i in range(32)] + ['ymm' + str(i) + ',' for i in range(32)]) 
gen_regi = set(['r' + str(i) for i in range(8, 16)] + ['r' + str(i) + ',' for i in range(8, 16)] + ['r' + str(i) + 'b' for i in range(8, 16)] + ['r' + str(i) + 'b,' for i in range(8, 16)])
gen2_regi = set(['r' + str(i) + 'd' for i in range(8, 16)] + ['r' + str(i) + 'd,' for i in range(8, 16)] + ['r' + str(i) + 'w' for i in range(8, 16)] + ['r' + str(i) + 'w,' for i in range(8, 16)])
x86_regi_set = set()
x86_oprands = set(['cmpsb', 'cmpsw', 'cmpsd', 'retn', 'movsq', 'stosq', 'movsw', 'movsb', 'movsd', 'movsq' ,'scasb', 'cmpxchg'])
x86_regi_set = set.union(bounds_regi, debug_regi, control_regi, stack_regi, sse_regi, avx_regi, av2_regi, gen_regi, gen2_regi)
x86_directives = set(['extrn'])
instruction_mapping = defaultdict(set)
unique_instructions = set()
p_opcodes = set()

def write_line(line):
    # convert tokens to uppercase and join them with '~'
    out_line = '~'.join(token.upper() for token in line)
    out_line = re.sub(r'0RG_[0-9a-zA-Z]+', '<ARG>', out_line)
    out_line = out_line.replace("$", "-0")
    return out_line

def token_helper(token, function_names):    
    if token.startswith('var_'):
        return '<VAR>'   
    if token.startswith('arg_'):
        return '<ARG>'  
    if token in function_names:
        return '<FOO>'  
    if re.sub('\A[0-9A-Fa-f]+\Z', '0', token) == '0' or re.sub('[0-9A-Fa-f]+h', '0', token) == '0':
        return '0'  
    if token in x86_regi_set:
        return token 
    if token in x86_reg_re:
        return token    
    else:
        return '<TAG>'

def replace_labels(line, function_names):
    if line[0] == 'call':
        if line[1] == 'rax':
            return ['call', 'rax']
        if line[1].startswith('cs:'):
            return ['call', 'cs:<FOO>']
        return ['call', '<FOO>']
    
    for i, token in enumerate(line):
        # recording opcodes
        if i == 0:
            p_opcodes.add(token)
        # replace function names
        if token in function_names:
            if i == 0:
                continue
            line[i] = '<FOO>'
            continue
            
        if token == '=':
            if i == 0:
                return None
            line.remove(token)
            continue
        
        if token in x86_oprands:
            line[i] = token
            continue

        if line[0] in x86_directives:
            continue

        # replace string data type
        if token.startswith('_STR_'):
            if i == 0:
                return None
            line[i] = '<STRING>'
            continue
            
        # replace function names in the air
        if token.startswith('__') or token.startswith('_'):
            if i == 0:
                return None
            line[i] = '<F00>'
            continue
        
        # replace arg_abc types
        if token.startswith('arg_'):
            if i == 0:
                return None
            line[i] = '<ARG>'
            continue
            
        #replace var_abc types
        if token.startswith('var_'):
            if i == 0:
                return None
            line[i] = '<VAR>'
            continue           
            
        # replace offset
        if token == 'offset':
            line[i] = 'OFFSET'
            continue

        if token == 'short':
            line[i] = 'SHORT'
            continue
            
        if token == 'jmp':
            line[i] = 'JMP'
            continue
            
        if token == 'leave':
            line[i] = 'LEAVE'
            continue
        
        if token == 'proc':
            return None
            
        if token == 'near':
            return None
            
        if token == 'endp':
            return None
        
        # hex or decimal
        if i != 0 and (re.sub('\A[0-9A-Fa-f]+\Z', '0', token) == '0' or re.sub('[0-9A-Fa-f]+h', '0', token) == '0'):
            line[i] = '0'
            continue
        
        # replace segment registers
        # "segment register : offset register"; A segment register is one of CS, DS, ES, FS, GS, or SS) 
        if token.startswith('cs:'):
            if i == 0:
                return None 
            
            temp = token[3:]
            if temp.startswith('dword_'):
                temp = 'dword_<ADDR>'
            elif temp.startswith('qword_'):
                temp = 'qword_<ADDR>'
            else:
                temp = '<ADDR>'
            line[i] = 'cs:' + temp
            if token[-1] == ',':
                line[i] += ','
            continue
        if token.startswith('ds:'):
            if i == 0:
                return None 
            temp = token[3:]
            if temp.startswith('dword_'):
                temp = 'dword_<ADDR>'
            elif temp.startswith('qword_'):
                temp = 'qword_<ADDR>'
            else:
                temp = '<ADDR>'
            line[i] = 'ds:' + temp
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('fs:'):
            if i == 0:
                return None 
            temp = token[3:]
            if temp.startswith('dword_'):
                temp = 'dword_<ADDR>'
            elif temp.startswith('qword_'):
                temp = 'qword_<ADDR>'
            else:
                temp = '<ADDR>'
            line[i] = 'fs:' + temp
            if token[-1] == ',':
                line[i] += ','
            continue
            
        # replace data types
        if token == '<DWORD_PTR>':
            if i == 0:
                return None
            continue
        
        if token == '<QWORD_PTR>':
            if i == 0:
                return None
            continue
        
        if token == '<TBYTE_PTR>':
            if i == 0:
                return None
            continue

        if token == '<BYTE_PTR>':
            if i == 0:
                return None
            continue
        
        if token == '<WORD_PTR>':
            if i == 0:
                return None
            continue
        
        if token == '<XMMWORD_PTR>':
            if i == 0:
                return None
            continue

        if token == '<YMMWORD_PTR>':
            if i == 0:
                return None
            continue

        if token == '<ZMMWORD_PTR>':
            if i == 0:
                return None
            continue

        # replace dummy name prefixes from ida
        # this has to be the last step of the whole filtering process
        # refer to https://hex-rays.com/blog/igors-tip-of-the-week-34-dummy-names/
        if token.startswith('sub_'):
            if i == 0:
                return None 
            line[i] = 'sub_<FOO>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('def_'):
            if i == 0:
                return None 
            line[i] = 'DEF_<TAG>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('locret_'):
            if i == 0:
                return None 
            line[i] = 'locret_<TAG>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('loc_'):
            if i == 0:
                return None 
            line[i] = 'loc_<TAG>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('off_'):
            if i == 0:
                return None 
            line[i] = 'off_<OFFSET>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('seg_'):
            if i == 0:
                return None 
            line[i] = 'seg_<ADDR>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('asc_'):
            if i == 0:
                return None 
            line[i] = 'asc_<STR>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('byte_'):
            if i == 0:
                return None 
            line[i] = 'byte_<BYTE>'
            if token[-1] == ',':
                line[i] += ','
            continue
        
        
        if token.startswith('word_'):
            if i == 0:
                return None 
            line[i] = 'word_<WORD>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('dword_'):
            if i == 0:
                return None 
            line[i] = 'DWORD_<WORD>'
            if token[-1] == ',':
                line[i] += ','
            continue
        
        if token.startswith('qword_'):
            if i == 0:
                return None 
            line[i] = 'qword_<WORD>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('byte3_'):
            if i == 0:
                return None 
            line[i] = 'byte3_<BYTE>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('xmmword_'):
            if i == 0:
                return None 
            line[i] = 'xmmword_<WORD>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('ymmword_'):
            if i == 0:
                return None 
            line[i] = 'ymmword_<WORD>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('packreal_'):
            if i == 0:
                return None 
            line[i] = 'packreal_<BIT>'
            if token[-1] == ',':
                line[i] += ','
            continue
        
        if token.startswith('flt_'):
            if i == 0:
                return None 
            line[i] = 'flt_<BIT>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('dbl_'):
            if i == 0:
                return None 
            line[i] = 'dbl_<BIT>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('tbyte_'):
            if i == 0:
                return None 
            line[i] = 'tbyte_<BYTE>'
            if token[-1] == ',':
                line[i] += ','
            continue
        
        if token.startswith('stru_'):
            if i == 0:
                return None 
            line[i] = 'stru_<TAG>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('custdata_'):
            if i == 0:
                return None 
            line[i] = 'custdata_<TAG>'
            if token[-1] == ',':
                line[i] += ','
            continue
            
        if token.startswith('algn_'):
            if i == 0:
                return None 
            line[i] = 'algn_<TAG>'
            if token[-1] == ',':
                line[i] += ','
            continue
        
        if token.startswith('unk_'):
            if i == 0:
                return None 
            line[i] = 'unk_<TAG>'
            if token[-1] == ',':
                line[i] += ','
            continue
        
        # starting with opcode 
        if i == 0:
            continue
        
        if token in x86_regi_set:
            continue
            
        if token in x86_reg_re:
            continue
        
        # [rax + 8]
        if token[0] == '[':
            if token[-1] == ',':
                comma = True
                token = token[:-1]
            else:
                comma = False                    
            # remove the '[]'
            temp = token[1: -1]
            
            # mark the punctuation
            symbol = '+'
            if '-' in temp:
                symbol = '-'
            
            # split by symbol
            items = temp.split(symbol)
            for idx, item in enumerate(items):                
                # search for asterisk * 
                if '*' in item:
                    sub_items = item.split('*')
                    
                    for idx_s, s in enumerate(sub_items):
                        sub_items[idx_s] = token_helper(s, function_names)
                    
                    items[idx] = '*'.join(sub_items)
                    continue
                
                items[idx] = token_helper(item, function_names)                     
                
            line[i] = '[' + symbol.join(items) + ']'
            if comma:
                line[i] += ','
            continue
        
        # NO MATCH. Shouldn't be there
        else:  
            line[i] = '<TAG>'
            if token[-1] == ',':
                line[i] += ','
    return line

def edge_helper(line):
    """Extract edge information from a line."""
    words = line.split()
    return int(words[3].split('"')[1]), int(words[5].split('"')[1])

def preprocessing(input_file, output_file):
    with open(input_file, 'r') as fp_in, open(output_file, 'w') as fp_out:
        lines = fp_in.readlines()
        node = False 
        cur_labels = []
        cur_bb = ""
        cur_edges = [] # save the edges
        for line in lines:
            ori = line
            ending = False
            if not node and line.startswith('node: {'):
                if line.strip().endswith('}'):
                    continue
                node = True
                continue
            elif not node and line.startswith('edge: {'):
                sourcename, targetname = edge_helper(line)
                cur_edges.append([sourcename, targetname])
                continue
            elif not node:
                continue
            elif node and line.strip().endswith('}'):
                ending = True
                node = False

            if ';' in line:
                    line = line.split(';')
                    line = line[0]
            if '#' in line:
                line = line.split('#')
                line = line[0]
            if '"' in line:
                line = line.split('"')
                line = line[0]


            # split line
            if 'xmmword ptr' in line:
                line = line.replace('xmmword ptr', '<XMMWORD_PTR>')
            if 'ymmword ptr' in line:
                line = line.replace('ymmword ptr', '<YMMWORD_PTR>')
            if 'zmmword ptr' in line:
                line = line.replace('zmmword ptr', '<ZMMWORD_PTR>')
            if 'tbyte ptr' in line:
                line = line.replace('tbyte ptr', '<TBYTE_PTR>')
            if 'qword ptr' in line:
                line = line.replace('qword ptr', '<QWORD_PTR>')
            if 'dword ptr' in line:
                line = line.replace('dword ptr', '<DWORD_PTR>')
            if 'byte ptr' in line:
                line = line.replace('byte ptr', '<BYTE_PTR>') 
            if 'word ptr' in line:
                line = line.replace('word ptr', '<WORD_PTR>')  
            
            # replace " - " with '-'
            line = line.replace(" - ", '-')
            line = line.split()

            if line[0] in x86_directives:
                continue

            if line[0] == 'endbr64' or line[0] == 'dq':
                continue
            
            # add lock 
            if line[0] == 'lock':
                lock = True 
                line = line[1:]
            else:
                lock = False
                
            if not line:
                continue
            line = replace_labels(line, function_names)

            if not line:
                continue

            line = write_line(line)
            if lock:
                line = "LOCK~" + line 
                
            cur_bb += line + ' '
            unique_instructions.add(line)
            instruction_mapping[line].add(ori)
            if ending:
                node = False
                cur_labels.append(cur_bb)
                cur_bb = ''
        cur_dict = dict()
        cur_dict['label'] = cur_labels
        cur_dict['graph'] = cur_edges
        json.dump(cur_dict, fp_out)    

