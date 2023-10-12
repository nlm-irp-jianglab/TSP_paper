#!/bin/bash
# Requirement: VIBRANT (https://github.com/AnantharamanLab/VIBRANT)
gid=$1
output_dir=$2
# ml hmmer prodigal gcc cmake
# export VIBRANT_DATA_PATH=/data/Irp-jiang/share/DB_Share/vibrant_db/
VIBRANT_run.py -i ${output_dir}/${gid}.fna -t 16

#### comment the following command lines if you want to keep the full results of VIBRANT ####
mkdir -p VIBRANT_${gid}_result
cp VIBRANT_${gid}/VIBRANT_results_${gid}/VIBRANT_integrated_prophage_coordinates_${gid}.tsv  VIBRANT_${gid}_result/
cp VIBRANT_${gid}/VIBRANT_phages_${gid}/GCA_000005845.2.phages_combined.fna  VIBRANT_${gid}_result/
cp VIBRANT_${gid}/VIBRANT_phages_${gid}/GCA_000005845.2.phages_combined.gbk  VIBRANT_${gid}_result/
rm -rf VIBRANT_${gid}