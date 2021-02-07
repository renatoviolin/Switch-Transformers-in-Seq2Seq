# Seq2Seq Switch Transformers

This repository implements Seq2Seq model using [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf) service.

The aim of this implementation is to confirm that this approach can be usefull even in smaller models size, producing better results with a little overhead on the computing time and with the same memory consuptiom, but with a model 3x times bigger than standard transformers.


# Project Details
For learning purpose I decided to not use any package with transformers implemented, so in this repository you find all the code to implement all steps of the standard transformer and the Switch Transformers:

The application is a Seq2Seq model to translate from EN to DE. This task is "easy" to a Transformer model, but the goal is to show how the Switch Transformers overfit faster the dataset.

The codes are inspired in [Bentrevett repository](https://github.com/bentrevett/pytorch-seq2seq) about Seq2Seq and [LabML](https://nn.labml.ai/transformers/switch/) about Switch Transformers. Those are amazing reference materials to this subject.

# Install

```
pip install -r requirements.text
python -m spacy download en
python -m spacy download de
``` 

# Running
For each experiment it will result in a file "results" with the loss to be ploted later.
```
python main.py
```
Plot the graph.
```
python plot_results.py
```
To change the model size, num_heads, num_experts take a look at [config.py](config.py).


# Results

Transformer model with the following parameters, all in the [config.py](config.py):
- Embedding dim: 512
- FF Hidden dim: 512
- Layers: 3
- Heads: 8
- Max Seq Len: 50
- Batch Size: 256

| Model | # Parameters | GPU Memory | Time per epoch
| --- | --- | --- | --- |
| Standard Transformer | 18,000,653 | 4918 MB | [00:05<00:00,  6.71it/s]
| Switch Transformers (16)  | 65,327,981  | 5596 MB |[00:10<00:00,  3.80it/s]

<img src=img/loss.jpg>
<img src=img/memory.jpg>


# References
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf)
- [Yannic Kilcher video on Switch Transformers](https://www.youtube.com/watch?v=iAR8LkkMMIM)
- [Bentrevett repository](https://github.com/bentrevett/pytorch-seq2seq)
- [LabML](https://nn.labml.ai/transformers/switch/)