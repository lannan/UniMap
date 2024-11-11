import os
import shutil

# define file extensions to be cleaned up
CLEANUP_EXTENSIONS = {'.i64', '.id2', '.idb', '.id0', '.id1', '.nam', '.til'}

def is_cleanup(name):
    return any(name.endswith(ext) for ext in CLEANUP_EXTENSIONS)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_input_files(input_path, output_path, log_path):
    input_files = [f for f in os.listdir(input_path) if f not in {'gen_gdl_func.py', 'funcname_file.txt'}]

    for file_name in input_files:
        target_file = os.path.join(input_path, file_name)
        log_file = os.path.join(log_path, f"{file_name}.txt")
        
        # choose the appropriate IDA command based on architecture
        ida_command = f"ida64 -c -A -Sgen_gdl_func.py -L{log_file} {target_file}"
        os.system(ida_command)

        # post-process the generated files
        cleanup_and_move_files(input_path, output_path)

def cleanup_and_move_files(input_path, output_path):
    for name in os.listdir(input_path):
        source_file = os.path.join(input_path, name)
        destination_file = os.path.join(output_path, name)

        if is_cleanup(name):
            os.remove(source_file)
        elif name.endswith('.gdl'):
            os.rename(source_file, destination_file)
        elif name == 'funcname_file.txt':
            shutil.copy(source_file, destination_file)

def main():
    arch = 'x86'
    input_path = f'./binary/{arch}/'
    output_path = f'./gdl/{arch}/'  
    log_path = './logs'
    create_directory(output_path)
    create_directory(log_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist")

    process_input_files(input_path, output_path, log_path)

if __name__ == "__main__":
    main()

