import idautils
import pickle
import ida_funcs

def load_func_names():
    """Load function names from a file if it exists, otherwise return an empty set."""
    try:
        with open('funcname_file.txt', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return set()

def save_func_names(func_names):
    """Save function names to a file."""
    with open("funcname_file.txt", "wb") as f:
        pickle.dump(func_names, f)

def process_functions(gdl_path, func_names):
    """Process functions to extract line numbers and generate flow graphs."""
    for func in idautils.Functions():
        pfn = ida_funcs.get_func(func)
        if not pfn or not ida_funcs.is_func_entry(pfn):
            continue

        clean_register_renames(pfn)

        func_name = idc.get_func_name(func)
        if func == idaapi.BADADDR or not func_name:
            print("Break from loop due to bad address or empty function name.")
            break

        func_names.add(func_name)
        func_path = f"{gdl_path}@{func_name}"
        f0 = idaapi.get_func(func)
        fc = idaapi.FlowChart(f0, flags=idaapi.FC_PREDS)

        extract_line_numbers(fc, pfn, func_path, func_name)

def clean_register_renames(pfn):
    """Delete register rename information for a function."""
    for rv in reversed(pfn.regvars):
        try:
            idaapi.del_regvar(pfn, rv.start_ea, rv.end_ea, rv.canon)
        except Exception as e:
            print(f"Failed to delete register variable {rv.canon}: {e}")

def extract_line_numbers(fc, pfn, func_path, func_name):
    """Extract start and end line numbers for each basic block and set comments."""
    found = False
    for block in fc:
        ea0, ea1 = block.start_ea, block.end_ea
        ea2 = idc.prev_head(ea1)
        start_linnum = ida_nalt.get_source_linnum(ea0)
        end_linnum = ida_nalt.get_source_linnum(ea2)

        if start_linnum == idaapi.BADADDR:
            continue

        clean_block_register_renames(pfn, ea0, ea1)

        # Search for the valid end line number
        while end_linnum == idaapi.BADADDR and ea2 > ea0:
            ea2 = idc.prev_head(ea2)
            end_linnum = ida_nalt.get_source_linnum(ea2)

        found = True
        ida_bytes.set_cmt(ea0, f"start_linnum: {start_linnum}", 0)
        ida_bytes.set_cmt(ea2, f"end_linnum: {end_linnum}", 0)

        print(f"Start line number: {start_linnum}")
        print(f"End line number: {end_linnum}")
        print("-" * 24)

    if found:
        idaapi.gen_flow_graph(func_path, func_name, pfn, 0, 0, idaapi.CHART_PRINT_NAMES | idaapi.CHART_GEN_GDL)

def clean_block_register_renames(pfn, ea0, ea1):
    """Clean register rename information within a specific block range."""
    for rv in pfn.regvars:
        try:
            idaapi.del_regvar(pfn, ea0, ea1, rv.canon)
        except Exception as e:
            print(f"Failed to delete register variable {rv.canon} in block: {e}")

def main():
    ida_auto.auto_wait()
    gdl_path = idc.get_idb_path().replace(".idb", "")

    func_names = load_func_names()
    process_functions(gdl_path, func_names)
    save_func_names(func_names)

    if idaapi.cvar.batch:
        print("All done, exiting.")
        ida_pro.qexit(0)

if __name__ == "__main__":
    main()
