# Files description:
- assign_serotype2TSP.py: the Python script to perform the 
- \<species\>.cl602sero_full.tsv: The original files of tailspike protein clusters corresponding to the bacterial serotypes in each species. Each file contains five columns: tailspike protein cluster ID at 60% identity (1st), bacteria serotype (2nd), the number of vOTUs in the cluster with the serotype (3rd), the number of vOTUs in the cluster (4th), percentage of vOTUs in the cluster with the serotype (5th).
- results/\<species\>.cl602sero_filtered.tsv: The filtered associations for the tailspike protein clusters with bacterial serotypes in each species. Each file contains seven columns: tailspike protein cluster ID at 60% identity (1st), bacteria serotype (2nd), species (3rd), the number of vOTUs in the cluster with the serotype (4th), the number of vOTUs in the cluster (5th), percentage of vOTUs in the cluster with the serotype (6th), the confidence level for the association (7th).

# To run the scripts:
```
mkdir -p results
for species in Pseudomonas_aeruginosa Salmonella Escherichia_coli_Shigella Acinetobacter Klebsiella
do
    echo "Processing species: $species"
    cut -f1 ${species}.cl602sero_full.tsv | sort | uniq | parallel -k python assign_serotype2TSP.py {} ${i} > results/${species}.cl602sero_filtered.tsv
done
```
