# Large-scale Genomic Survey with Deep Learning-based Method Reveals Strain-Level Phage Specificity Determinants

This repository contains the code to 1) find prophage regions from bacterial genomes and 2) assign serotypes to tailspike protein clusters implementation for the research paper titled "Large-scale Genomic Survey with Deep Learning-based Method Reveals Strain-Level Phage Specificity Determinants". 

## Prerequisite
- python >= 3.8
- pandas

## File structure
```
├── assign_serotype
│   ├── Acinetobacter.cl602sero_full.tsv
│   ├── assign_serotype2TSP.py
│   ├── Escherichia_coli_Shigella.cl602sero_full.tsv
│   ├── Klebsiella.cl602sero_full.tsv
│   ├── Pseudomonas_aeruginosa.cl602sero_full.tsv
│   ├── README
│   ├── results
│   └── Salmonella.cl602sero_full.tsv
├── data 
│   └── TSP_ids_and_clusters.txt
├── find_prophage_regions
│   ├── bacteria_assembly_id.tsv
│   ├── MD5SUM.txt
│   ├── README
│   ├── step1.download_genome.sh
│   ├── step2.run_VIBRANT.sh
│   └── VIBRANT_prophage_regions.tsv
├── LICENSE
└── README.md

```

## Running scripts
Please read the individual README files in find_prophage_regions/ and assign_serotype/ for detailed instructions.

## Citation
If you use SpikeHunter in your research, please cite it as follows: Yang Y, Dufault-Thompson K, Yan W, Cai T, Xie L, Jiang X. Deciphering Phage-Host Specificity Based on the Association of Phage Depolymerases and Bacterial Surface Glycan with Deep Learning. bioRxiv. 2023:2023-06.
