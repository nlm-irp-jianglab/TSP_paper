mkdir -p results
for i in Pseudomonas_aeruginosa Salmonella Escherichia_coli_Shigella Acinetobacter Klebsiella
do
    echo "Processing species: $i"
    cut -f1 ${i}.cl602sero_full.tsv |sort |uniq | parallel -k python assign_serotype2TSP.py {} ${i} > results/${i}.cl602sero_filtered.tsv
done