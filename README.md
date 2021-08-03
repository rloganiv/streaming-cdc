Benchmarking Scalable Methods for Streaming Cross Document Entity Coreference
===

This repository contains the code for the paper:

Logan IV, Robert L., et al. "Benchmarking Scalable Methods for Streaming Cross Document Entity Coreference." ACL 2021.


## Getting Started

To get started, we recommend creating a fresh python virtual environment and
then running:
`pip install -e .`
to install the `meercat` library and its dependencies.

Note: MEERCAT was the original name for this project. Feel free to look at the
project history to see what it stands for!


## Downloading the Benchmark Data

Preprocessed data and licenses for MedMentions and Zeshel are available on
Google drive, and can be accessed via the following links:
- [MedMentions](https://drive.google.com/drive/folders/1zrd4Q-b0K2BMHIi5RtC-cqmt9Mz7S_rz?usp=sharing)
- [Zeshel](https://drive.google.com/drive/folders/1c-uyruQQLDAbtnGLi3YSEw5G2_GamlYc?usp=sharing)

Because data for the AIDA benchmark relies on the Reuters Corpus, you will need
to follow the [instructions for obtaining this
dataset](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads), and then run the following script to preprocess it into the correct format:
```
python scripts/preprocess_aida_yago2.py --input [PATH TO AIDA-YAGO2-dataset.tsv] --prefix [DIRECTORY TO SAVE JSONL FILES TO]
```

Note: We've also made our code for preprocessing MedMentions and Zeshel
available in the `scripts/` directory.
However, due to updates to PubMed the output of the MedMentions preprocessing
script is inconsistent, and so evaluation should be performed using the
preprocessed data linked above.


## Baselines & Evaluation

TO BE COMPLETED


## Citation

```
@inproceedings{logan-iv-etal-2021-benchmarking,
    title = "Benchmarking Scalable Methods for Streaming Cross Document Entity Coreference",
    author = "Logan IV, Robert L  and
      McCallum, Andrew  and
      Singh, Sameer  and
      Bikel, Dan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.364",
    doi = "10.18653/v1/2021.acl-long.364",
    pages = "4717--4731",
    abstract = "Streaming cross document entity coreference (CDC) systems disambiguate mentions of named entities in a scalable manner via incremental clustering. Unlike other approaches for named entity disambiguation (e.g., entity linking), streaming CDC allows for the disambiguation of entities that are unknown at inference time. Thus, it is well-suited for processing streams of data where new entities are frequently introduced. Despite these benefits, this task is currently difficult to study, as existing approaches are either evaluated on datasets that are no longer available, or omit other crucial details needed to ensure fair comparison. In this work, we address this issue by compiling a large benchmark adapted from existing free datasets, and performing a comprehensive evaluation of a number of novel and existing baseline models. We investigate: how to best encode mentions, which clustering algorithms are most effective for grouping mentions, how models transfer to different domains, and how bounding the number of mentions tracked during inference impacts performance. Our results show that the relative performance of neural and feature-based mention encoders varies across different domains, and in most cases the best performance is achieved using a combination of both approaches. We also find that performance is minimally impacted by limiting the number of tracked mentions.",
}
```

## Disclaimer

The code and data in this repository was written and collected by Robert L.
Logan IV as a student at UC Irvine, not as an intern at Google.
