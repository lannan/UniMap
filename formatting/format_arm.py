import time
import os
import pickle
import re
import json
from collections import deque

arm_directives = set(['CODE16', 'CODE32', 'EXPORT', 'ALIGN', 'DCD', 'nptr', 'DCB', 'IMPORT'])
arm_addrs = set(['errnum', 'name', 's'])
gen_regi = set(['R' + str(i) for i in range(0, 16)] + ['R' + str(i) + '!' for i in range(0, 16)])
map_regi = {'SP': 'R13', 'LR': 'R14', 'PC': 'R15'}
var_regi = set(['V' + str(i) for i in range(1, 9)])
arg_regi = set(['A' + str(i) for i in range(1, 5)])
float_regi = set(['D' + str(i) for i in range(0, 32)] + ['d' + str(i) for i in range(0, 32)] + ['S' + str(i) for i in range(0, 32)] + ['s' + str(i) for i in range(0, 32)] + ['Q' + str(i) for i in range(0, 32)] + ['q' + str(i) for i in range(0, 32)]) 
cp15_regi = set(['C' + str(i) for i in range(0, 16)] + ['P' + str(i) for i in range(0, 16)] + ['c' + str(i) for i in range(0, 16)] + ['p' + str(i) for i in range(0, 16)])
program_regi = set(['CPSR', 'SPSR'])
other_regi = set(['SB', 'IP', 'SP', 'LR', 'PC'])
arm_regi_set = set.union(gen_regi, var_regi, arg_regi, float_regi, cp15_regi, program_regi, other_regi)
arm_oprands = set(['GE', 'CC', 'LT', 'GT', 'FPSCR', 'LS', 'HI', 'LE', 'CS', 'SY', 'MI', 'PL', 'SY', 'MI', 'WEAK'])
unknown_opcodes = set(['GE', 'CC', 'LT', 'GT', 'WEAK'])


def write_line(line):
    # convert tokens to uppercase and join them with '~'
    out_line = '~'.join(token.upper() for token in line)
    out_line = re.sub(r'0RG_[0-9a-zA-Z]+', '<ARG>', out_line)
    out_line = out_line.replace("$", "-0")
    return out_line
        
def replace_dummy(token):
        if token.startswith('_STR_'):
            return '<STRING>'
                
        if token.startswith('sub_'):
            return 'sub_<FOO>'

        if token.startswith('locret_'):
            return 'locret_<TAG>'
        
        if token.startswith('def_'):
            return 'def_<TAG>'

        if token.startswith('loc_'):
            return 'loc_<TAG>'

        if token.startswith('off_'):
            return 'off_<OFFSET>'
            
        if token.startswith('seg_'):
            return 'seg_<ADDR>'
        
        if token.startswith('asc_'):
            return 'asc_<STR>'

        if token.startswith('byte_'):
            return 'byte_<BYTE>'

        if token.startswith('word_'):
            return 'word_<WORD>'

        if token.startswith('dword_'):
            return 'DWORD_<WORD>'

        if token.startswith('qword_'):
            return 'qword_<WORD>'

        if token.startswith('byte3_'):
            return 'byte3_<BYTE>'

        if token.startswith('xmmword_'):
            return 'xmmword_<WORD>'

        if token.startswith('ymmword_'):
            return 'ymmword_<WORD>'

        if token.startswith('packreal_'):
            return 'packreal_<BIT>'

        if token.startswith('flt_'):
            return 'flt_<BIT>'

        if token.startswith('dbl_'):
            return 'dbl_<BIT>'

        if token.startswith('tbyte_'):
            return 'tbyte_<BYTE>'

        if token.startswith('stru_'):
            return 'stru_<TAG>'

        if token.startswith('custdata_'):
            return 'custdata_<TAG>'

        if token.startswith('algn_'):
            return 'algn_<TAG>'

        if token.startswith('unk_'):
            return 'unk_<TAG>'
        
        return None

def helper(cur_token, function_names):
    res = ''
    zero = False
    cur_puncs = deque(re.findall(r'[+-]', cur_token))
    
    # split by '+' or '-' and process each part
    subs = re.split(r'\+|\-', cur_token)
    for sub in subs:
        sub = sub.strip()
        par = sub.startswith('(')
        if par:
            sub = sub[1:]

        if sub.startswith('#'):
            sub = sub[1:]

        if sub.isdigit() or sub.startswith('0x'):
            res += '0'
            zero = True
        elif sub in function_names:
            res += '<FOO>'
        elif sub in arm_addrs:
            res += '<ADDR>'
        elif sub.startswith('var_'):
            res += '<VAR>'
        elif sub.startswith('arg_'):
            res += '<ARG>'
        elif replace_dummy(sub):
            res += replace_dummy(sub)
        elif zero:
            res += '<ADDR>'
        else:
            res += '<TAG>'
        if cur_puncs:
            res += cur_puncs.popleft()
    if par:
        res += ')'
    return res


def replace_labels(line, function_names):
    for i, token in enumerate(line):
        if i == 0:
            if token.endswith('.W'):
                line[0] = line[0][:-2]
                continue
        if token == '=':
            if i == 0:
                return None
            continue
        
        if token[-1] == ',':
            token = token[:-1]
            comma = True
        else:
            comma = False
       
        if token[0] == '#':
            if i == 0:
                return None
            token = token[1:]
        
        if token[0] == '=':
            equal_sign = True
            token = token[1:]
        else:
            equal_sign = False

        if token[0] == '-':
            minus_sign = True
            token = token[1:]
        else:
            minus_sign = False
        token = re.sub('#(0x[0-9a-fA-F]+)', '0', token)
        token = re.sub('#-(0x[0-9a-fA-F]+)', '$', token)
        token = re.sub('#([0-9a-fA-F]+)', '0', token)
        token = re.sub('#-([0-9a-fA-F]+)', '$', token)
        token = re.sub('([+-]?[0-9]+\.[0-9e]+[+-]?[0-9]+)', '0', token)
        token = re.sub('([+-]?[0-9]+\.+[0-9]+)', '0', token)

        if token in arm_oprands:
            line[i] = token
            continue

        if token == 'NE' or token == 'EQ':
            continue
        
        if token in arm_directives:
            return None
        
        if token in function_names:
            if i == 0:
                return None
            line[i] = "<FOO>"
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
        
        if token.startswith(':upper16:'):
            if i == 0:
                return None
            line[i] = 'UPPER<ADDR>'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
        
        if token.startswith(':lower16:'):
            if i == 0:
                return None
            line[i] = 'LOWER<ADDR>'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
        
        if token.startswith("arg_"):
            if i == 0:
                return None
            line[i] = '<ARG>'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
            
        if token.startswith("var_"):
            if i == 0:
                return None
            line[i] = '<VAR>'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
            
        if token.startswith('jpt_'):
            if i == 0:
                return None
            line[i] = 'JPT<ADDR>'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
            
        dummy = replace_dummy(token)
        if dummy:
            if i == 0:
                return None
            line[i] = dummy
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
           
        if token.isdigit() or token.startswith('0x') or token == 'NaN' or token == '+Inf':
            if i == 0:
                return None
            line[i] = '0'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
        
        if token[0] == '(' and token[-1] == ')':
            if i == 0:
                return None
            item = token[1:-1]
            if '+' or '-' in item:
                line[i] = helper(item, function_names)
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
            
        
        if token[0] == '[':
            if i == 0:
                return None
            exclamation = False
            suffix = None
            if token[-1] == '!':
                items = token[1:-2].split(',')
                exclamation = True
            elif token[-1] == ']':
                items = token[1:-1].split(',')  
            else:
                items = token[1:].split(']')[0].split(',')
                suffix = token[1:].split(']')[1]
            cur = '[' 
            for item in items:
                if item in arm_regi_set:
                    cur += item
                elif item[0] == '-' and item[1:] in arm_regi_set:
                    cur += item
                elif item == '0':
                    cur += '0'
                elif item == '$':
                    cur += '$'
                elif '+' in item or '-' in item:
                    cur += helper(item, function_names)
                elif item.startswith('#'):
                    cur += '0'
                elif '0' in item or '$' in item: 
                    cur += item
                else:
                    cur += '<TAG>'
                cur += ','
            cur = cur[:-1]
            if exclamation:
                cur += ']!'
            else:
                cur += ']'
            line[i] = cur
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if suffix:
                line[i] += suffix
            if comma:
                line[i] += ','
            continue
 
        if token in arm_regi_set:
            continue
        
        if token[0] == '-1' and token [1:] in arm_regi_set:
            continue
            
        if i == 0:
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','  
            continue
        
        if ',' in token:
            line[i] = token
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','  
            continue
            
        if '#' in token:
            idx = token.index('#')
            line[i] = token[:idx] + '0' 
            
        else:
            line[i] = '<TAG>'
            
        if equal_sign:
            line[i] = "=" + line[i]
        if minus_sign:
            line[i] = '-' + line[i]
        if comma:
            line[i] += ','   
    return line  

def stack_operation(line):
    if line[-1][-1] == '^':
        line[-1] = line[-1][:-1]
    opcode = line[0]
    if line[1][-2] == '!':
        exclamation = '!'
    else:
        exclamation = ''
    sec = line[1][:-1].replace('!','')

    punc = '+'
    if opcode.startswith('S'):
        punc = '-'
    elif opcode.startswith('L'):
        punc = '+'
    else:
        unknown_opcodes.add(opcode)
    
    register_list = []
    result = []
    
    cur_registers = line[2][1:-1].split(',')
    for cur_regi in cur_registers:
        if '-' in cur_regi:
            cur_range = cur_regi.split('-')  
            for cur_idx, cur_item in enumerate(cur_range):
                if cur_item in map_regi:
                    cur_range[cur_idx] = map_regi[cur_item]

            char_regi = cur_range[0][0]
            start, end = int(cur_range[0][1:]), int(cur_range[1][1:])
            for val in range(start, end + 1):
                register_list.append(char_regi + str(val))
        else:
            register_list.append(cur_regi)
        
    if punc == '-':
        register_list = register_list[::-1] # reverse

    for regi_item in register_list:
        result.append("{}~[{}{}0]{},{}".format(opcode, sec, punc, exclamation, regi_item))
    return result

def split_registers(token):
    cur_registers = token.split(',')
    register_list = []
    for cur_regi in cur_registers:
        if '-' in cur_regi:
            cur_range = cur_regi.split('-')
            # replace sp, lr, pc with r13 - r15
            for cur_idx, cur_item in enumerate(cur_range):
                if cur_item in map_regi:
                    cur_range[cur_idx] = map_regi[cur_item]                    
            char_regi = cur_range[0][0]
            start, end = int(cur_range[0][1:]), int(cur_range[1][1:])
            for val in range(start, end + 1):
                register_list.append(char_regi + str(val))
        else:
            register_list.append(cur_regi)
    return register_list

def replace_range(line, cur_bb):
    ori = line
    line = line.split()
    opcode = line[0]
    if opcode.endswith('.W'):
        line[0] = line[0][:-2]
    
    if opcode.startswith('PUSH'):
        rev = True
    else:
        rev = False
    if len(line) == 2:
        register_list = split_registers(line[1][1:-1])
        if rev:
            register_list.reverse()
        
        for regi_item in register_list:
            written_line = line[0] + '~' + regi_item.upper()
            written_line = re.sub('0RG_[0-9a-zA-Z]+', '<ARG>', written_line)
            written_line = written_line.replace('$', '-0')
            cur_bb += written_line + ' '
        
    elif len(line) == 3:
        cur_lines = stack_operation(line)
        for cur_line in cur_lines:
            written_line = cur_line.upper()
            written_line = re.sub('0RG_[0-9a-zA-Z]+', '<ARG>', written_line)
            written_line = written_line.replace('$', '-0')
            cur_bb += written_line + ' '


    elif len(line) == 4 and line[2].startswith('{'):
        idx_end = line[2].index('}')
        cur_remain = line[2][idx_end + 1:]

        register_list = split_registers(line[2][1:idx_end])
        for regi_item in register_list:
            written_line = line[0] + '~' + line[1] + regi_item.upper() + cur_remain + line[3]
            written_line = re.sub('0RG_[0-9a-zA-Z]+', '<ARG>', written_line)
            written_line = written_line.replace('$', '-0')
            cur_bb += written_line + ' '
    else:
        print("LENGTH DOES NOT MATCH:   ", ori)
    return cur_bb

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
            if (not node) and line.startswith('node: { title'):
                if line.strip().endswith('}'):
                    continue
                node = True
                continue
            elif not node and line.startswith('edge: { '):
                sourcename, targetname = edge_helper(line)
                cur_edges.append([sourcename, targetname])
                continue
            elif not node:
                continue
            elif node and line.strip().endswith(' }'):
                ending = True
                node = False
                
            if ';' in line: # text after ";" are comments generated from ida. ignore(remove) the comments
                    line = line.split(';')
                    line = line[0]
            if '"' in line:
                line = line.split('"')
                line = line[0]             
            if " = " in line or "%" in line:
                continue
            if line.startswith('VST1') or line.startswith('VST4'):
                line = line.split()
                written_line = write_line(line)
                cur_bb += written_line + ' '
            elif '{' in line:
                if len(line.split()) == 5 and line.split()[4].startswith("{"):
                    line = line.split()[:4] + ['0']
                    formatted_line = replace_labels(line, function_names)
                    if not formatted_line:
                        continue
                    written_line = write_line(formatted_line)
                    cur_bb += written_line + ' '
                else:
                    cur_bb = replace_range(line, cur_bb)
            else:
                line = line.replace(" - ", '-')
                line = line.split()
                if line[0] in arm_directives:
                    continue
                formatted_line = replace_labels(line, function_names)
                if not formatted_line:
                    continue
                written_line = write_line(formatted_line)
                cur_bb += written_line + ' '
            if ending:
                node = False
                cur_labels.append(cur_bb)
                cur_bb = ""
        cur_dict = dict()
        cur_dict['label'] = cur_labels
        cur_dict['graph'] = cur_edges
        json.dump(cur_dict, fp_out) 

def edge_helper(line):
    words = line.split()
    return int(words[3].split('"')[1]), int(words[5].split('"')[1])
