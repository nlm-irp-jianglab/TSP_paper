# Deciphering Phage-Host Specificity Based on the Association of Phage Depolymerases and Bacterial Surface Glycan using Deep Learning

This repository contains the code to assign serotypes to tailspike protein clusters implementation for the research paper titled "Deciphering Phage-Host Specificity Based on the Association of Phage Depolymerases and Bacterial Surface Glycan using Deep Learning." 

## Prerequisite
- python >= 3.8
- pandas

## Running
```
cd assign_serotype/
mkdir -p results
for species in Pseudomonas_aeruginosa Salmonella Escherichia_coli_Shigella Acinetobacter Klebsiella
do
    echo "Processing species: $species"
    cut -f1 ${species}.cl602sero_full.tsv |sort |uniq | parallel -k python assign_serotype2TSP.py {} ${i} > results/${species}.cl602sero_filtered.tsv
done
```

## Citation
If you use SpikeHunter in your research, please cite it as follows: Yang Y, Dufault-Thompson K, Yan W, Cai T, Xie L, Jiang X. Deciphering Phage-Host Specificity Based on the Association of Phage Depolymerases and Bacterial Surface Glycan with Deep Learning. bioRxiv. 2023:2023-06.
