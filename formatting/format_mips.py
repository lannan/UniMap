import time
import os
import pickle
import re
import json
from collections import defaultdict

mips_directives = set(['.DATA','.TEXT','.EXTERN','.GLOBL','.ALIGN','.ASCII','.ASCIIZ','.BYTE','.HALF','SPACE','.WORD'])
assembler_regi = set(['$AT'])
special_regi = set(['$PC','$HI','$LO', '$W0'])
map_regi = {'$ZERO': 'R0', '$AT': 'R1', '$GP': 'R28', '$SP': 'R29', '$FP': 'R30', '$RA': 'R31',
            '$V0': 'R2', '$V1': 'R3', '$A0': 'R4', '$A1': 'R5', '$A2': 'R6', '$A3': 'R7',
            '$T0': 'R8', '$T1': 'R9', '$T2': 'R10', '$T3': 'R11', '$T4': 'R12', '$T5': 'R13', '$T6': 'R14', '$T7': 'R15', '$T8': 'R24', '$T9': 'R25', 
            '$S0': 'R16', '$S1': 'R17', '$S2': 'R18', '$S3': 'R19', '$S4': 'R20', '$S5': 'R21', '$S6': 'R22', '$S7': 'R23',
            '$K0': 'R26', '$K1': 'R27', '$GP': 'R28', '$SP': 'R29', '$FP': 'R30', '$RA': 'R31'}

opcode = set(['ADD', 'ADDI', 'ADDU', 'ADDIU', 'AND', 'ANDI', 'LUI', 'NOR', 'OR', 'ORI', 'SLT', 'SLTI', 'SLTIU', 'SLTU', 'SUB', 'SUBU', 'XOR', 'XORI'] + 
             ['SLL', 'SLLV', 'SRA', 'SRAV', 'SRL', 'SRLV'] + ['DIV', 'DIVU', 'MFHI', 'MTHI', 'MTLO', 'MULT', 'MULTU'] + 
             ['BEQ', 'BGEZ', 'BGEZAL', 'BGTZ', 'BLEZ', 'BLTZ', 'BLTZAL', 'BNE', 'BREAK', 'J', 'JAL', 'JALR', 'JR', 'MFC0', 'MTC0', 'SYSCALL'] +
             ['LB', 'LBU', 'LH', 'LBU', 'LW', 'SB', 'SH', 'SW', 'LI'])
op_map = {'BEQZ': 'BEQ', 'BNEZ': 'BNE', 'SWL':'SW', 'SWR':'SW', 'LWC1': 'LW', 'LWL': 'LW', 'LWR': 'LW', 'LBU': 'LB', 
         'LHU': 'LH', 'MADDU':'MADD', 'MULU': 'MUL', 'ADDU': 'ADD', 'ADDI': 'ADD', 'DIVU':'DIV', 'SLTIU':'SLT', 'SUBU': 'SUB', 
          'SLTU': 'SLT', 'XORI': 'XOR', 'ANDI': 'AND', 'SLTI': 'SLT', 'MULTU':'MULT', 'ADDIU':'ADD', 'SRAV':'SRA', 'SLLV':'SLL', 'SRLV':'SRL'}

# floating point registers: https://www.cs.unibo.it/~solmi/teaching/arch_2002-2003/AssemblyLanguageProgDoc.pdf PG. 24
# http://www.jaist.ac.jp/iscenter-/mpc/old-machines/altix3700/opt/toolworks/totalview.6.3.0-1/doc/html/ref_guide/MIPSFloatingPointRegisters.html
float_regi = set(['$F' + str(i) for i in range(0, 31)] + ['$f' + str(i) for i in range(0, 31)] + ['$FCSR'])
float_cond_regi = set(['$FCC' + str(i) for i in range(0, 8)] + ['$fcc' + str(i) for i in range(0, 8)])
mips_regi_set = set.union(assembler_regi, float_regi, float_cond_regi, map_regi, special_regi)


instruction_mapping = defaultdict(set)
unique_instructions = set()
p_opcodes = set()

def write_line(line):
    # convert tokens to uppercase and join them with '~'
    out_line = '~'.join(token.upper() for token in line)
    out_line = re.sub(r'0RG_[0-9a-zA-Z]+', '<ARG>', out_line)
    out_line = out_line.replace("$", "-0")
    return out_line

def replace_dummy(token, function_names):
   # replace dummy name prefixes from ida
        # this has to be the last step of the whole filtering process
        # refer to https://hex-rays.com/blog/igors-tip-of-the-week-34-dummy-names/
        # replace string data type
        if token.startswith('_STR_'):
            return '<STRING>'
                
        if token.startswith('VAR_'):
            return '<VAR>'

        if token.startswith("ARG_"):
            return '<ARG>'
            
        if token.startswith('JPT_'):
            return 'JPT_<ADDR>'

        if token.startswith('SUB_'):
            return 'SUB_<TAG>'

        if token.startswith('LOCRET_'):
            return 'LOC_<TAG>'
        
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
    if len(line) == 1:
        return line
    for i, token in enumerate(line):
        # opcode
        if i == 0:                
            p_opcodes.add(token)
            if token.startswith('MOV'):
                token = 'MOV'
            elif token.endswith('.S') or token.endswith('.D') or token.endswith('.W'):
                token = token[:-2]
            
            if token in op_map:
                token = op_map[token]
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
        
        # replace function names:
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

        if token[0] == '(' and token[-1] == ')':
            if token.count('(') > 1 and token.count(')') > 1:
                # get the src register name
                token = token.split(')(')[-1][:-1]
                if token in map_regi:
                    line[i] = '<OFF>' + map_regi[token]
                elif token in mips_regi_set:
                    line[i] = '<OFF>' + token[1:]
                else:
                    continue
                if equal_sign:
                    line[i] = "=" + line[i]
                if minus_sign:
                    line[i] = '-' + line[i]
                if comma:
                    line[i] += ','
                p_opcodes.add(line[0])
            else:
                line[i] = '<OFF>'
                if equal_sign:
                    line[i] = "=" + line[i]
                if minus_sign:
                    line[i] = '-' + line[i]
                if comma:
                    line[i] += ','
                p_opcodes.add(line[0])
            continue
            
        if token[-1] == ')' and '(' in token:
            subs = token[:-1].split('(')
            sub = subs[-1]
            if sub in map_regi:
                cur_regi = map_regi[sub]
            elif sub in mips_regi_set:
                cur_regi = sub[1:]
            else:
                print("CANNOT FOUND REGI second if ", token)
                
            line[i] = "<OFF>" + cur_regi
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
                    
        if token.isdigit() or token.startswith('0X'):
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
            if comma:
                line[i] += ','
            continue  
        
        # replace function names:
        if token.startswith('_STR_'):
            if i == 0:
                return None
            line[i] = "<STRING>"
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
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
            if comma:
                line[i] += ','
            continue
            
        if token.startswith("VAR_"):
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
            
        if token.startswith('JPT_'):
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
            
        # registers
        if token in map_regi:
            line[i] = map_regi[token]
            if equal_sign:
                line[i] = "=" + line[i]
            if minus_sign:
                line[i] = '-' + line[i]
            if comma:
                line[i] += ','
            continue
        if token in mips_regi_set:
            line[i] = token[1:]
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
        else:
            line[i] = '<TAG>'
            
        if equal_sign:
            line[i] = "=" + line[i]
        if minus_sign:
            line[i] = '-' + line[i]
        if comma:
            line[i] += ','   
    return line  

def edge_helper(line):
    words = line.split()
    return int(words[3].split('"')[1]), int(words[5].split('"')[1])

def preprocessing(input_file, output_file, arch='mips'):
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
            # comments are tagged with # in mips assembly idapro
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

            if line[0] in mips_directives:
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
        
