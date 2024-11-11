# Training MAIE
--------
1. Train OAIE using [fasttext](https://github.com/facebookresearch/fastText) for each architecture. 
2. Run map_embedding.py to generate Multi-architecture Instruction Embeddings. For example: 
```bash
python3 map_embeddings.py --cuda --log logfile.txt mips.vec x86.vec mips.emb x86.emb
```

The code is modified based on [VecMap](https://github.com/artetxem/vecmap), an open-source implementation for learning cross-lingual word embedding mappings, as described in:

Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
