# Files description:
- bacteria_assembly_id.tsv: The NCBI assembly IDs for bacterial genomes. It contains two columns: the genome assembly ID (Genome) and the species of the bacteria (Species).
- VIBRANT_prophage_regions.tsv.tar.gz: The prophage fragments dectected from host bacterial genomes listed in bacteria_assembly_id.tsv which are organized from the original results from VIBRANT. It contains five columns: The genome assembly ID where the prophage is derived from (Genome), The scaffold of the genome where the prophage is derived from (Scaffold), The unique name for the prophage fragment (Prophage fragment), the nucleotide start position of the prophage in the genome (Nucleotide start) and the nucleotide end position of the prophage in the genome (Nucleotide end). You could get this file by running scripts `step1.download_genome.sh` and `step2.run_VIBRANT.sh` for all the genomes listed in `bacteria_assembly_id.tsv` and merging all the results. 
- step1.download_genome.sh: the bash script to download the bacterial genomes from NCBI.
- step2.run_VIBRANT.sh: the bash script to run VIBRANT on the bacterial genomes to find prophage regions.
- MD5SUM.txt: the file to record the md5sum values for the files listed as above.

# Requirements:
- seqkit (https://github.com/shenwei356/seqkit)
- VIBRANT (https://github.com/AnantharamanLab/VIBRANT)

# To run the scripts:
```
# Create a folder to store the results
mkdir -p bacterial_genomes
data_folder=bacterial_genomes

# Step 1. Download bacterial genomes.
# 1) download in batch
cut -f1 bacteria_assembly_id.tsv | parallel -k ./step1.download_genome.sh {} ${data_folder}
# 2) download a single genome of interest
gID=GCA_000005845.2 # for example
./step1.download_genome.sh ${gid} ${data_folder}

# Step 2. run ViBRANT
# 1) run in batch
cut -f1 bacteria_assembly_id.tsv | parallel -k ./step2.run_VIBRANT.sh {} ${data_folder}
# 2) run VIBRANT on a single genome of interest
gID=GCA_000005845.2 # for example
./step2.run_VIBRANT.sh ${gid} ${data_folder}

# The final result is stored in the folder VIBRANT_${gid}_result/, which includes the .fna, .gbk files and coordinates information for the prophage regions. Other files are deleted to save space. If you want to keep the full results of VIBRANT, please comment (i.e. adding "#" in front of each line) the following command lines in step2.run_VIBRANT.sh：

mkdir -p VIBRANT_${gid}_result
cp VIBRANT_${gid}/VIBRANT_results_${gid}/VIBRANT_integrated_prophage_coordinates_${gid}.tsv  VIBRANT_${gid}_result/
cp VIBRANT_${gid}/VIBRANT_phages_${gid}/GCA_000005845.2.phages_combined.fna  VIBRANT_${gid}_result/
cp VIBRANT_${gid}/VIBRANT_phages_${gid}/GCA_000005845.2.phages_combined.gbk  VIBRANT_${gid}_result/
rm -rf VIBRANT_${gid}

```
