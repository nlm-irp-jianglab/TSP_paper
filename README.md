# Large-scale Genomic Survey with Deep Learning-based Method Reveals Strain-Level Phage Specificity Determinants

This repository contains the code to 1) find prophage regions from bacterial genomes and 2) assign serotypes to tailspike protein clusters implementation for the research paper titled "Large-scale Genomic Survey with Deep Learning-based Method Reveals Strain-Level Phage Specificity Determinants". 

## Prerequisite
- python >= 3.8
- pandas

## File structure
```
├── ablation_studies
│   ├── add_dropout_layers
│   ├── best_models
│   ├── delete_hidden_layer
│   ├── README.md
│   └── SeqVec_encoder
├── assign_serotype
│   ├── Acinetobacter.cl602sero_full.tsv
│   ├── assign_serotype2TSP.py
│   ├── Escherichia_coli_Shigella.cl602sero_full.tsv
│   ├── Klebsiella.cl602sero_full.tsv
│   ├── Pseudomonas_aeruginosa.cl602sero_full.tsv
│   ├── README.md
│   ├── results
│   └── Salmonella.cl602sero_full.tsv
├── data
│   ├── bacteria_genomes
│   └── TSP_ids_and_clusters.txt
├── find_prophage_regions
│   ├── bacteria_assembly_id.tsv
│   ├── MD5SUM.txt
│   ├── README.md
│   ├── step1.download_genome.sh
│   ├── step2.run_VIBRANT.sh
│   └── VIBRANT_prophage_regions.tsv.tar.gz
├── LICENSE
└── README.md
```

## File description
- data/TSP_ids_and_clusters.txt: The file contains tailspike protein IDs idenitified in this study with their corresponding clusters at various protein identities.
- The description for other files can be found in the README.md files in individual folders.

## Running scripts
Please read the individual README.md files in find_prophage_regions/ and assign_serotype/ for detailed instructions.

## Citation
If you use SpikeHunter in your research, please cite it as follows: Yang Y, Dufault-Thompson K, Yan W, Cai T, Xie L, Jiang X. Deciphering Phage-Host Specificity Based on the Association of Phage Depolymerases and Bacterial Surface Glycan with Deep Learning. bioRxiv. 2023:2023-06.
