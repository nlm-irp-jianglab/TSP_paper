#!/bin/bash
# Requirement: seqkit (https://github.com/shenwei356/seqkit)
gid=$1
output_dir=$2
rm -f ${gid}.zip
curl -OJX GET "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/${gid}/download?include_annotation_type=GENOME_FASTA&filename=${gid}.zip" -H "Accept: application/zip"
unzip ${gid}.zip -d ${gid}
mv ${gid}/ncbi_dataset/data/${gid}/*_genomic.fna ${output_dir}/${gid}.fna.tmp
seqkit fx2tab ${output_dir}/${gid}.fna.tmp | awk -F"\t" -v gid=${gid} '{print ">"gid"_"NR"\n"$2}' > ${output_dir}/${gid}.fna
rm ${output_dir}/${gid}.fna.tmp
rm -f ${gid}.zip
rm -rf ${gid}