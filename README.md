# Unimap

## 1. Installation 

Main libraries:

- [Python 3](https://www.python.org/downloads/) 
- [FastText](https://fasttext.cc/) 
- [NumPy](https://pypi.org/project/numpy/)
- [SciPy](https://github.com/scipy/scipy)
- [CuPy](https://cupy.dev/) (optional, only required for CUDA support)
- [PyTorch](https://pytorch.org/) （for the downstream tasks)

## 1. Generate CFG
We use IDAPro to generate the Control Flow Graphs. The script can be found [here](./cfg/gen_linum.py). 




## 1. Formatting
To generate the formatted GDL files, run the preprocessing function using the architecture-specific script: ```preprocessing(input_file, output_file)```. 

Available formatting scripts: 

- [x86](./formatting/format_x86.py)
- [ARM](./formatting/format_arm.py)
- [MIPS](./formatting/format_mips.py)
- [PPC](./formatting/format_ppc.py)


## 2. Train Multi-architecture Instruction Embeddings
Pre-trained embeddings are available for download here: [MAIE Embeddings](https://drive.google.com/drive/folders/1KPyV3uhQVnblpBBHH6hwp-GwMZoDQKzt?usp=drive_link).
The training scripts can be found in the [MAIE](./maie) sub-directory. 

## 3. Downstream Tasks
Function Similarity detection and malware detection tasks can be found [Here](./tasks)


## 4. Acknowledgements
Thanks for [FastText](https://github.com/facebookresearch/fastText), [VecMap](https://github.com/artetxem/vecmap), and [FuncGNN](https://github.com/aravi11/funcGNN) papers for making their code public. 

## 5. Citation
If you find the code helpful, please cite this work:
```
@inproceedings{wang2023can,
  title={Can a Deep Learning Model for One Architecture Be Used for Others?$\{$Retargeted-Architecture$\}$ Binary Code Analysis},
  author={Wang, Junzhe and Sharp, Matthew and Wu, Chuxiong and Zeng, Qiang and Luo, Lannan},
  booktitle={32nd USENIX Security Symposium (USENIX Security 23)},
  pages={7339--7356},
  year={2023}
}
```
