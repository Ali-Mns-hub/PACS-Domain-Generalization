# PACS Dataset & DomainBed Protocol

This directory is designated for the **PACS** dataset, used here to evaluate Domain Generalization techniques.

## DomainBed Batching Protocol

Unlike standard classification tasks where mini-batches are sampled randomly from the entire dataset, Domain Generalization often requires structured sampling. 

In this repository, we follow a data loading protocol similar to **DomainBed**. For each training iteration (step), a single mini-batch is constructed by concatenating smaller mini-batches sampled evenly from each of the available *source domains*. This ensures the model sees a balanced representation of diverse domains at every optimization step, which is crucial for techniques like ERM and MIRO.

## Expected Directory Structure

Before running the notebooks, ensure the PACS dataset is extracted here. The structure must be exactly as follows:

```
data/
├── art_painting/
│   ├── dog/
│   ├── elephant/
│   ├── ...
├── cartoon/
│   ├── dog/
│   ├── ...
├── photo/
│   ├── dog/
│   ├── ...
└── sketch/
    ├── dog/
    ├── ...
```

* Ensure that all 7 classes exist within each of the 4 domain folders.
* The notebook uses `scikit-learn`'s `train_test_split` alongside custom PyTorch DataLoaders to handle the DomainBed-style sampling strategy.
