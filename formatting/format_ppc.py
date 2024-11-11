import time
import os
import pickle
import re
import json
from collections import defaultdict

gen_regi = set(['R' + str(i) for i in range(0, 32)] + ['F' + str(i) for i in range(0, 32)])
v_regi = set(['V' + str(i) for i in range(0, 32)] + ['VRSAVE'] + ['VS' + str(i) for i in range(0, 64)]) # vrsave: non-volatile 32-bit registers
# general-purpose registers(GPRS), Floating-Point Registers (FPRs), conditional registers (CRs)
other_regi = set(['GPR' + str(i) for i in range(0, 32)] + ['FPR' + str(i) for i in range(0, 32)] + ['CR' + str(i) for i in range(0, 8)])
ppc_regi_set = set.union(gen_regi, other_regi, v_regi)
ppc_directives = set(['.EXTERN'])
ppc_op_map = {'ADDI': 'ADD', 'XORI': 'XOR', 'SLWI': 'SLW'}


def write_line(line):
    # convert tokens to uppercase and join them with '~'
    out_line = '~'.join(token.upper() for token in line)
    out_line = re.sub(r'0RG_[0-9a-zA-Z]+', '<ARG>', out_line)
    out_line = out_line.replace("$", "-0")
    return out_line

def replace_dummy(token, function_names):
        if token.startswith('_STR_'):
            return '<STRING>'
        if token.startswith('VAR_'):
            return '<VAR>'
        if token.startswith("ARG_"):
            return '<ARG>'
        if token.startswith('JPT_'):
            return 'JPT_<ADDR>'
        if token.startswith('SUB_'):
            return 'SUB_<FOO>'
        if token.startswith('LOCRET_'):
            return 'LOCRET_<TAG>'
        if token.startswith('DEF_'):
            return 'DEF_<TAG>'
        if token.startswith('LOC_'):
            return 'LOC_<TAG>'
        if token.startswith('OFF_'):
            return 'OFF_<OFFSET>'
        if token.startswith('SEG_'):
            return 'SEG_<ADDR>'
        if token.startswith('ASC_'):
            return 'ASC_<STR>'
        if token.startswith('BYTE_'):
            return 'BYTE_<BYTE>'
        if token.startswith('WORD_'):
            return 'WORD_<WORD>'
        if token.startswith('DWORD_'):
            return 'DWORD_<WORD>'
        if token.startswith('QWORD_'):
            return 'QWORD_<WORD>'
        if token.startswith('BYTE3_'):
            return 'BYTE3_<BYTE>'
        if token.startswith('XMMWORD_'):
            return 'XMMWORD_<WORD>'
        if token.startswith('YMMWORD_'):
            return 'YMMWORD_<WORD>'
        if token.startswith('PACKREAL_'):
            return 'PACKREAL_<BIT>'
        if token.startswith('FLT_'):
            return 'FLT_<BIT>'
        if token.startswith('DBL_'):
            return 'DBL_<BIT>'
        if token.startswith('TBYTE_'):
            return 'TBYTE_<BYTE>'
        if token.startswith('STRU_'):
            return 'STRU_<TAG>'
        if token.startswith('CUSTDATA_'):
            return 'CUSTDATA_<TAG>'
        if token.startswith('ALGN_'):
            return 'ALGN_<TAG>'
        if token.startswith('UNK_'):
            return 'UNK_<TAG>'
        if token.startswith('OPTARG'):
            return 'OPTARG_<ADDR>'            
        return None

def replace_labels(line, function_names):
#     if len(line) == 1:
#         return line
    for i, token in enumerate(line):
        # opcode
        if i == 0:
            p_opcodes.add(token)
            # [.] dot suffix is used to enable the update of CR
            # [+] A + or - sign following a branch mnemonic sets the instruction's branch prediction flag
            if token.endswith('.') or token.endswith('+') or token.endswith('-'):
                token = token[:-1]
            if token in ppc_op_map:
                token = ppc_op_map[token]
            line[0] = token
            continue 
        if token == '=':
            if i == 0:
                return None
            continue
        
        if token == ';':
            if i == 0:
                return None
            else:
                line = line[:i]
                break
        # tag the comma
        if token[-1] == ',':
            token = token[:-1]
            comma = True
        else:
            comma = False
        if token[0] == '#':
            if i == 0:
                return None
            token = token[1:]
            
        # mark and remove @ha and @lo from the current token      
        if '@HA' in token:
            token = token.replace("@HA", "")
            athigh = True
        else:
            athigh = False
        if '@L' in token:
            token = token.replace("@L", "")
            atlow = True
        else:
            atlow = False
        
        # replace '#-' with '$'
        token = re.sub('#(0X[0-9a-fA-F ]+)', '0', token)
        token = re.sub('#([0-9a-fA-F ]+)', '0', token)
        if token[0] == '=':
            equal_sign = True
            token = token[1:]
        else:
            equal_sign = False
        if token[0] == '-':
#             minus_sign = True
            minus_sign = False
            token = token[1:]
        else:
            minus_sign = False           
        if token == 'NE' or token == 'EQ':
            continue
    
        # cror 0*cr7+eq, 4*cr7+gt, 4*cr7+eq
        if '+EQ' in token or '+GT' in token or '+LT' in token or '+SO' in token:
            if '*' in token:
                token = '0*' + token.split('*')[1]
            line[i] = token
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'
            if comma:
                line[i] += ','
            continue 
        
        # replace function names:
        if token in function_names:
            if i == 0:
                return None
            line[i] = "<FOO>"
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'
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
            if athigh:
                line[i] = line[i] + '@ha'
            if atlow:
                line[i] = line[i] + '@l'
            if comma:
                line[i] += ','
            continue
        
        if token.startswith('_STR_'):
            if i == 0:
                return None
            line[i] = "<STRING>"
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'
            if comma:
                line[i] += ','
            continue
            
        if token.startswith("ARG_"):
            if i == 0:
                return None
            line[i] = '<ARG>'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'   
            if comma:
                line[i] += ','
        
        if token.startswith("VAR_"):
            if i == 0:
                return None
            line[i] = '<VAR>'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'
            if comma:
                line[i] += ','
            continue
            
        if token.startswith('JPT_'):
            if i == 0:
                return None
            line[i] = 'JPT<ADDR>'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'
            if comma:
                line[i] += ','
            continue
            
        # dummy generated by IDA
        dummy = replace_dummy(token, function_names)
        if dummy:
            if i == 0:
                return None
            line[i] = dummy
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'
            if comma:
                line[i] += ','
            continue
        
        if token[0] == '(' and token[-1] == ')':
            if token.count('(') > 1 and token.count(')') > 1:
                # get the src register name
                token = token.split(')(')[-1][:-1]
                if token in ppc_regi_set:
                    line[i] = '<OFF>' + token
                else:
                    continue
                if equal_sign:
                    line[i] = "=" + line[i]
                if minus_sign:
                    line[i] = '-' + line[i]
                if athigh:
                    line[i] = line[i] + '@HA'
                if atlow:
                    line[i] = line[i] + '@L'
                if comma:
                    line[i] += ','
            else:           
                line[i] = '<OFF>'
                if equal_sign:
                    line[i] = "=" + line[i]
                if minus_sign:
                    line[i] = '-' + line[i]
                if comma:
                    line[i] += ','
            continue
            
        if token[-1] == ')' and '(' in token:
            subs = token[:-1].split('(')
            sub = subs[-1]
            if sub in ppc_regi_set:
                cur_regi = sub
            else:
                print("CANNOT FOUND REGI", token)
                
            line[i] = "<OFF>" + cur_regi
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'
            if comma:
                line[i] += ','
            continue
        
        # registers
        if token in ppc_regi_set:
            continue
        if token[0] == '-1' and token [1:] in ppc_regi_set:
            continue        
        if ',' in token:
            # rlwinm    r9, r29, 0,18,19
            subs = token.split(',')
            res = []
            for sub in subs:
                if sub.isdigit():
                    res.append('0')
                else:
                    res.append('<TAG>')
            line[i] = ','.join(res)
            
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'
            if comma:
                line[i] += ','  
            continue   
            
        # immediate operands
        if token.isdigit() or token.startswith('0X'):
            if i == 0:
                return None
            line[i] = '0'
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if athigh:
                line[i] = line[i] + '@HA'
            if atlow:
                line[i] = line[i] + '@L'
            if comma:
                line[i] += ','
            continue   
        else:
            line[i] = '<TAG>'
            
        if equal_sign:
            line[i] = "=" + line[i]
        if minus_sign:
            line[i] = '-' + line[i]
        if athigh:
                line[i] = line[i] + '@HA'
        if atlow:
            line[i] = line[i] + '@L'
        if comma:
            line[i] += ','   
    return line  

def edge_helper(line):
    words = line.split()
    return int(words[3].split('"')[1]), int(words[5].split('"')[1])

def preprocessing(input_file, output_file, arch='ppc'):
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

            # replace " - " with '-'
            line = line.upper()
            line = line.replace(" - ", '-')
            line = line.split()

            if line[0] in ppc_directives:
                continue

            line = replace_labels(line, function_names)
            if not line:
                continue
            line = write_line(line)

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
        