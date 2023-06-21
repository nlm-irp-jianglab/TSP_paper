########################################################################
# This script is used to assign a serotype to a given TSP cluster (60%)
########################################################################

# Input: 1) TSP cluster id; 2) species
# output: serotype(s) to the TSP cluster
# Details should be listed here.

import sys
import pandas as pd

clstr_id = sys.argv[1]
species = sys.argv[2]

tolerance_cutoff = 0.1

def perc2label(hit, perc):
    if (perc > 0.9 and hit > 3) or (perc==1.0 and hit <= 3):
        return "Highly confident"
    elif (perc > 0.1 and hit > 3) or (perc==1.0 and hit <= 3):
        return "Confident"
    else:
        return "Uncertain"

clstr_lv = clstr_id.split("_")[0]
infile = species+"."+clstr_lv+"2sero_full.tsv"
df = pd.read_csv(infile, names=["clstr_id", "serotype", "hit_count", "total_count", "percentage"], sep="\t")

hit_dict = {}
for index, row in df.iterrows():
    if row["clstr_id"] == clstr_id:
        hit_dict[str(row["serotype"])] = [row["hit_count"], row["total_count"], row["percentage"]]

sorted_hit_dict = sorted(hit_dict.items(), key=lambda x:x[1][2], reverse=True)

final_sero = []
final_hit_ct = []
final_tot_ct = []
final_perc = []

K_sero = []
K_hit_ct = []
K_tot_ct = []
K_perc = []

nonK_sero = []
nonK_hit_ct = []
nonK_tot_ct = []
nonK_perc = []

if species in ["Pseudomonas_aeruginosa", "Salmonella"]:
    # only have O-antigen
    max_hit = sorted_hit_dict[0]
    max_sero = max_hit[0]
    max_hit_perc = max_hit[1][2]
    for ele in sorted_hit_dict:
        if abs(ele[1][2] - max_hit_perc) <= tolerance_cutoff:
            nonK_sero.append(ele[0])
            nonK_hit_ct.append(ele[1][0])
            nonK_tot_ct.append(ele[1][1])
            nonK_perc.append(ele[1][2])
            
    final_sero = nonK_sero
    final_hit_ct = nonK_hit_ct
    final_tot_ct = nonK_tot_ct
    final_perc = nonK_perc

elif species in ["Acinetobacter", "Klebsiella"]:
    # Rationale: K >> O/OC
    # get K first. If no K-antigen or K-antigen is poor, search for O/OC-antigen
    K_dict = [ele for ele in sorted_hit_dict if ele[0].startswith("K")]
    if len(K_dict) !=0:
        max_hit = K_dict[0]
        max_sero = max_hit[0]
        max_hit_perc = max_hit[1][2]
        for ele in K_dict:
            if abs(ele[1][2] - max_hit_perc) <= tolerance_cutoff:
                K_sero.append(ele[0])
                K_hit_ct.append(ele[1][0])
                K_tot_ct.append(ele[1][1])
                K_perc.append(ele[1][2])
        
    nonK_dict = [ele for ele in sorted_hit_dict if not ele[0].startswith("K")]
    if len(nonK_dict) != 0:
        max_hit = nonK_dict[0]
        max_sero = max_hit[0]
        max_hit_perc = max_hit[1][2]
        for ele in nonK_dict:
            if abs(ele[1][2] - max_hit_perc) <= tolerance_cutoff:
                nonK_sero.append(ele[0])
                nonK_hit_ct.append(ele[1][0])
                nonK_tot_ct.append(ele[1][1])
                nonK_perc.append(ele[1][2])
                
    if len(nonK_perc) !=0:
        if len(K_perc)!=0:
            if len(K_perc)>=5:
                # if too many K_perc, choose nonK
                final_sero = nonK_sero
                final_hit_ct = nonK_hit_ct
                final_tot_ct = nonK_tot_ct
                final_perc = nonK_perc
            else:
                # if max K and nonK is not significantly different, use K instead of nonK
                diff = K_perc[0] - nonK_perc[0]
                if diff < 0 and abs(diff) <= 0.25:
                    final_sero = K_sero
                    final_hit_ct = K_hit_ct
                    final_tot_ct = K_tot_ct
                    final_perc = K_perc
                # if max K > nonK, use K instead of nonK
                elif diff >= 0:
                    final_sero = K_sero
                    final_hit_ct = K_hit_ct
                    final_tot_ct = K_tot_ct
                    final_perc = K_perc
                else:
                    final_sero = nonK_sero
                    final_hit_ct = nonK_hit_ct
                    final_tot_ct = nonK_tot_ct
                    final_perc = nonK_perc
        else:
            # len(K_perc)==0
            final_sero = nonK_sero
            final_hit_ct = nonK_hit_ct
            final_tot_ct = nonK_tot_ct
            final_perc = nonK_perc
    else:
        # len(nonK_perc) ==0
        final_sero = K_sero
        final_hit_ct = K_hit_ct
        final_tot_ct = K_tot_ct
        final_perc = K_perc
        
elif species == "Escherichia_coli_Shigella":
    # Rationale: Klebsiella-K >= Ecoli-K > O > Others
    K_dict = [ele for ele in sorted_hit_dict if ele[0].startswith("K") or ele[0].startswith("Escherichia_coli-K")]
    if len(K_dict) != 0:
        max_hit = K_dict[0]
        max_sero = max_hit[0]
        max_hit_perc = max_hit[1][2]
        for ele in K_dict:
            if abs(ele[1][2] - max_hit_perc) <= tolerance_cutoff:
                K_sero.append(ele[0])
                K_hit_ct.append(ele[1][0])
                K_tot_ct.append(ele[1][1])
                K_perc.append(ele[1][2])

    nonK_dict = [ele for ele in sorted_hit_dict if not ele[0].startswith("K") and not ele[0].startswith("Escherichia_coli-K")]
    if len(nonK_dict) != 0:
        max_hit = nonK_dict[0]
        max_sero = max_hit[0]
        max_hit_perc = max_hit[1][2]
        for ele in nonK_dict:
            if abs(ele[1][2] - max_hit_perc) <= tolerance_cutoff:
                nonK_sero.append(ele[0])
                nonK_hit_ct.append(ele[1][0])
                nonK_tot_ct.append(ele[1][1])
                nonK_perc.append(ele[1][2])
                
    if len(nonK_perc) !=0:
        if len(K_perc)!=0:
            if len(K_perc)>=5:
                # if too many K_perc, choose nonK
                final_sero = nonK_sero
                final_hit_ct = nonK_hit_ct
                final_tot_ct = nonK_tot_ct
                final_perc = nonK_perc
            else:
                # if max K and nonK is not significantly different, use K instead of nonK
                diff = K_perc[0] - nonK_perc[0]
                if diff < 0 and abs(diff) <= 0.25:
                    final_sero = K_sero
                    final_hit_ct = K_hit_ct
                    final_tot_ct = K_tot_ct
                    final_perc = K_perc
                # if max K > nonK, use K instead of nonK
                elif diff >= 0:
                    final_sero = K_sero
                    final_hit_ct = K_hit_ct
                    final_tot_ct = K_tot_ct
                    final_perc = K_perc
                else:
                    if "O8/O9/O9a" in nonK_sero or "O89/O101/O162" in nonK_sero:
                        final_sero = K_sero
                        final_hit_ct = K_hit_ct
                        final_tot_ct = K_tot_ct
                        final_perc = K_perc
                    else:
                        final_sero = nonK_sero
                        final_hit_ct = nonK_hit_ct
                        final_tot_ct = nonK_tot_ct
                        final_perc = nonK_perc
        else:
            # len(K_perc)==0
            final_sero = nonK_sero
            final_hit_ct = nonK_hit_ct
            final_tot_ct = nonK_tot_ct
            final_perc = nonK_perc
    else:
        # len(nonK_perc) ==0
        final_sero = K_sero
        final_hit_ct = K_hit_ct
        final_tot_ct = K_tot_ct
        final_perc = K_perc

final_label = []

for index, perc in enumerate(final_perc):
    hit = final_hit_ct[index]
    final_label.append(perc2label(hit, perc))

if final_sero:
    zipped_lists = zip(final_sero, final_hit_ct, final_tot_ct, final_perc, final_label)
    sorted_zipped_lists = sorted(zipped_lists)

    final_sero, final_hit_ct, final_tot_ct, final_perc, final_label = zip(*sorted_zipped_lists)

    final_sero = ";".join(final_sero)
    final_hit_ct = ";".join(list(map(str, final_hit_ct)))
    final_tot_ct = ";".join(list(map(str, final_tot_ct)))
    final_perc = ";".join(list(map(str, final_perc)))
    final_label = ";".join(final_label)
    
    print("\t".join([clstr_id, species, final_sero, final_hit_ct, final_tot_ct, final_perc, final_label]))
else:
    print("\t".join([clstr_id, species, "-", "-", "-", "-", "Not_available"]))
