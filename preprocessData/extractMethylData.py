import pandas as pd
import pyBigWig
import pysam
import numpy as np
from multiprocessing import Pool, cpu_count
import os
shared_data_dir = "/media/desk16/share/zhiwei/"
result_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/result/"
human_TSS = pd.read_csv(result_dir+"gencode_v47_TSS_with_gene_unique.bed",sep="\t")
processed_data_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/processedData/"
# human_TSS = pd.read_csv(data_dir+"hg19_TSS.bed",sep="\t",header=None)

human_TSS = human_TSS[human_TSS.iloc[:,1] >= 2100]
fasta_path = shared_data_dir + "sequenceData/hg38.fa"

# Global variables (each process will initialize its own copy)
bw_meth = None
fasta = None

# Return a function that initializes file handles for each process
def make_initializer(bw_meth_path):
    def initializer():
        try:
            global bw_meth, bw_ctcf, fasta
            bw_meth = pyBigWig.open(bw_meth_path)
            fasta = pysam.FastaFile(fasta_path)
        except Exception as e:
            print(f"fail to open file：{e}")
            bw_meth = None
            fasta = None
    return initializer

# Function to extract per-region features: methylation, CTCF, sequence, etc.

def extract_region_data(row):
    """
    Extract methylation values, CTCF signal (all values), DNA sequence, and region length for a given genomic region.

    Parameters:
        row: list or tuple, containing [chrom, start, end]
        bw_meth: pyBigWig object for methylation
        bw_ctcf: pyBigWig object for CTCF
        fasta: pyfaidx.Fasta object

    Returns:
        list: [chrom, start, end, meth_values, meth_mean, ctcf_values, dmr_len, seq]
    """
    chrom, start, end = row[0], int(row[1]), int(row[2])
    
    try:
        # Methylation values
        meth_values = bw_meth.values(chrom, start, end, numpy=True)
        meth_values = np.nan_to_num(meth_values, nan=0.0)
        meth_mean = np.nanmean(meth_values)
        # DNA sequence
        seq = fasta.fetch(chrom, start, end).upper()

    except Exception as e:
        # Handle errors gracefully
        meth_values = np.array([])
        meth_mean = 0.0
        seq = ""
        # print(e)
        # optional: print(e) for debugging

    return [chrom, start, end, meth_values, meth_mean, seq]


if __name__ == "__main__":

    human_TSS.iloc[:,1] = human_TSS.iloc[:,1]-2000
    human_TSS.iloc[:,2] = human_TSS.iloc[:,2]+2000
    file_dir  = shared_data_dir+ "gptGeo/methyl/"
    file_list = os.listdir(file_dir)
    len(file_list)
    file_name =  file_list[0]
    processed_file_list = os.listdir(processed_data_dir)
    processed_file_names  = [el.split("_")[0]+".bigWig" for el in processed_file_list]
    file_list = [x for x in file_list if x not in processed_file_names]
    for file_name in file_list:
        print(f"Processing {file_name} ...")
        try:
            bw_meth_path = file_dir + file_name
            regions = human_TSS.iloc[:,:3].values.tolist()
            # Create multiprocessing pool; each pool processes one EID
            with Pool(processes=cpu_count(), initializer=make_initializer(bw_meth_path)) as pool:
                result_matrix = pool.map(extract_region_data, regions)
            valid_indices = [idx for idx, item in enumerate(result_matrix) if len(item[3]) > 0]
            valid_gene_name = human_TSS.iloc[:,3].values[valid_indices]
            meth_arrays = np.array([item[3] for item in result_matrix if len(item[3]) > 0])
            meth_values_df = pd.DataFrame(meth_arrays,valid_gene_name)
            human_TSS["tss"] = human_TSS.iloc[:,0].astype("str") +"_"+ human_TSS.iloc[:,1].astype("str")
            chr_pos = human_TSS["tss"].values[valid_indices]
            meth_values_df.insert(0, "chr_pos", chr_pos)
            meth_values_df.to_csv(processed_data_dir + "{}_methyl.csv".format(file_name.split(".")[0]))
        except Exception as e:
            # print(e)
            print(f"fail to preprocess {file_name}")



# 验证甲基化的均值和中位数
# meth_values_df["row_mean"] = meth_values_df.iloc[:,1:].mean(axis=1)
# meth_values_df["row_median"] = meth_values_df.iloc[:,1:].median(axis=1)
# np.mean(meth_values_df["row_mean"].values)
# np.median(meth_values_df["row_mean"].values)
            
