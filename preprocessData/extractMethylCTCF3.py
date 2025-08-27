import pandas as pd
import pyBigWig
import pysam
import numpy as np
from multiprocessing import Pool, cpu_count
import json

shared_data_dir =  "/media/desk16/share/zhiwei/"
result_dir = "/media/desk16/zhiwei/paper_code/MethylFormer_pre/result/"
fasta_path = shared_data_dir + "sequenceData/hg38.fa"
data_dir = "/media/desk16/share/zhiwei/sequenceData/"
saved_data_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/methylformer/data/"
human_TSS = pd.read_csv(result_dir+"gencode_v47_TSS_with_gene_unique.bed",sep="\t")
human_TSS = human_TSS[human_TSS.iloc[:,1] >= 2100]

# Global variables (each process will initialize its own copy)
bw_meth = None
bw_ctcf = None
fasta = None

# Return a function that initializes file handles for each process
def make_initializer(bw_meth_path, bw_ctcf_path):
    def initializer():
        global bw_meth, bw_ctcf, fasta
        bw_meth = pyBigWig.open(bw_meth_path)
        bw_ctcf = pyBigWig.open(bw_ctcf_path)
        fasta = pysam.FastaFile(fasta_path)
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
        meth_values = np.nan_to_num(meth_values, nan=0.0).tolist()
        # CTCF values
        ctcf_values = bw_ctcf.values(chrom, start, end, numpy=True)
        ctcf_values = np.nan_to_num(ctcf_values, nan=0.0).tolist()
        # DNA sequence
        seq = fasta.fetch(chrom, start, end).upper()

    except Exception as e:
        # Handle errors gracefully
        meth_values = []
        ctcf_values = []
        seq = ""
        # print(e) 
        print(chrom, start, end,e)   
    return [chrom, start, end, meth_values, ctcf_values, seq]

def make_dict_from_list(data, chr_list=None):
    keys = ["chr", "position", "end", "methyValue", "ctcfValue", "sequence"]
    result = []
    for item in data:
        # 如果 chr_list 为空就不过滤，否则只保留在 chr_list 中的染色体
        if len(item[3]) > 0 and (chr_list is None or item[0] in chr_list):
            result.append(dict(zip(keys, item)))
    return result

if __name__ == "__main__":
    # List of EIDs to process
    eid_li = ["ENCFF990DKJ"]
    eid_ctcf = {"ENCFF990DKJ":"ENCFF680NIF"}
    eid = eid_li[0]
    human_TSS.iloc[:,1] = human_TSS.iloc[:,1]-2000
    human_TSS.iloc[:,2] = human_TSS.iloc[:,2]+2000
    for eid in eid_li:
        print(f"Processing {eid} ...")
        try:
            # File paths for this eid
            bw_meth_path = shared_data_dir + f"gptGeo/ctcfMethyl/{eid}.bigWig"
            bw_ctcf_path = shared_data_dir + f"gptGeo/ctcfMethyl/{eid_ctcf[eid]}.bw"
            regions = human_TSS.iloc[:,:3].values.tolist()
            # Create multiprocessing pool; each pool processes one EID
            with Pool(processes=cpu_count(), initializer=make_initializer(bw_meth_path, bw_ctcf_path)) as pool:
                result_matrix = pool.map(extract_region_data, regions)

            json_result = make_dict_from_list(result_matrix)

            train_chr_list = [f"chr{i}" for i in range(1, 22)]
            train_data = make_dict_from_list(result_matrix, chr_list=train_chr_list)
            with open(saved_data_dir + "train_ctcf1.json", "w", encoding="utf-8") as f:
                json.dump(json_result, f, indent=4, ensure_ascii=False)
            # 验证集: chr22, chrX, chrY
            val_chr_list = ["chr22", "chrX", "chrY"]
            val_data = make_dict_from_list(result_matrix, chr_list=val_chr_list)
            with open(saved_data_dir + "valid_ctcf1.json", "w", encoding="utf-8") as f:
                json.dump(json_result, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed processing {eid}: {str(e)}")
            



