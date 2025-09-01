import pandas as pd
import pyBigWig
import pysam
import numpy as np
from multiprocessing import Pool, cpu_count
import json
import pandas as pd
shared_data_dir =  "/media/desk16/share/zhiwei/"
result_dir = "/media/desk16/zhiwei/paper_code/MethylFormer_pre/result/"
fasta_path = shared_data_dir + "sequenceData/hg38.fa"
data_dir = "/media/desk16/share/zhiwei/sequenceData/"
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

import numpy as np

def bin_signal(values, bin_size=40):
    """
    将信号按固定 bin_size 进行平均，并返回列表。
    
    Parameters:
        values: np.array, 信号序列
        bin_size: int, 每个 bin 的长度
    
    Returns:
        list: 每个 bin 的平均值
    """
    values = np.nan_to_num(values, nan=0.0)
    if len(values) == 0:
        return []

    n_bins = int(np.ceil(len(values) / bin_size))
    pad_len = n_bins * bin_size - len(values)
    if pad_len > 0:
        values = np.pad(values, (0, pad_len), mode='constant', constant_values=0)
    binned = values.reshape(n_bins, bin_size).mean(axis=1)
    return binned.tolist()

def extract_region_data(row, bin_size=40):
    """
    提取区域特征：binned methylation, binned CTCF, DNA sequence, region length
    """
    chrom, start, end = row[0], int(row[1]), int(row[2])
    try:
    # Methylation
        meth_values = bw_meth.values(chrom, start, end, numpy=True)
        meth_binned = bin_signal(meth_values, bin_size)
        # CTCF
        ctcf_values = bw_ctcf.values(chrom, start, end, numpy=True)
        ctcf_binned = bin_signal(ctcf_values, bin_size)

        # DNA sequence
        seq = fasta.fetch(chrom, start, end).upper()
    except Exception as e:
        meth_binned = []
        ctcf_binned = []
        seq = ""
        # print(e) 
        print(chrom, start, end,e)  
    return [chrom, start, end, meth_binned, ctcf_binned, seq]

def make_dict_from_list(data, chr_list=None):
    keys = ["chr", "position", "end", "methyValue", "ctcfValue", "sequence"]
    result = []
    for item in data:
        # 如果 chr_list 为空就不过滤，否则只保留在 chr_list 中的染色体
        if len(item[3]) > 0 and (chr_list is None or item[0] in chr_list):
            result.append(dict(zip(keys, item)))
    return result

def load_ctcf_wgbs_id(file_name="/media/desk16/zhiwei/paper_code/MethylFormer/preprocessData/data/ctcf_wgbs_mapping.csv"):
    # file_name = "/media/desk16/zhiwei/paper_code/MethylFormer/preprocessData/data/ctcf_wgbs_mapping.csv"
    ctcf_wgbs_id = pd.read_csv(file_name,sep="\t")
    ctcf_wgbs_id_dic = dict(zip(ctcf_wgbs_id["wgbs_id"], ctcf_wgbs_id["ctcf_id"]))
    return ctcf_wgbs_id_dic,ctcf_wgbs_id["wgbs_id"].values.tolist()[:5]

import os
import json

def merge_json_files(folder: str):
    """
    Merge JSON files in the given folder.
    - Files starting with 'train' will be merged into 'train_merged.json'
    - Files starting with 'valid' will be merged into 'valid_merged.json'
    """

    train_data = []
    valid_data = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if not filename.endswith(".json"):
            continue

        if filename.startswith("train"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    train_data.extend(data)
                else:
                    train_data.append(data)

        elif filename.startswith("valid"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    valid_data.extend(data)
                else:
                    valid_data.append(data)

    # Save merged results
    if train_data:
        with open(os.path.join(folder, "train_merged.json"), "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

    if valid_data:
        with open(os.path.join(folder, "valid_merged.json"), "w", encoding="utf-8") as f:
            json.dump(valid_data, f, ensure_ascii=False, indent=2)

    print("Merging completed!")

# Example usage:


if __name__ == "__main__":
    # List of EIDs to process
    bin_size = 20
    saved_data_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/methylformer/data/{}/".format(bin_size)
    os.makedirs(saved_data_dir, exist_ok=True)
    eid_li = ["ENCFF990DKJ"]
    eid_ctcf = {"ENCFF990DKJ":"ENCFF680NIF"}
    eid_ctcf,eid_li = load_ctcf_wgbs_id()
    
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

            # json_result = make_dict_from_list(result_matrix)
            train_chr_list = [f"chr{i}" for i in range(1, 22)]
            train_data = make_dict_from_list(result_matrix, chr_list=train_chr_list)
            with open(saved_data_dir + "train_ctcf_bin40_{}.json".format(eid), "w", encoding="utf-8") as f:
                json.dump(train_data, f, indent=4, ensure_ascii=False)
            # 验证集: chr22, chrX, chrY
            val_chr_list = ["chr22", "chrX", "chrY"]
            val_data = make_dict_from_list(result_matrix, chr_list=val_chr_list)
            with open(saved_data_dir + "valid_ctcf_bin40_{}.json".format(eid), "w", encoding="utf-8") as f:
                json.dump(val_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed processing {eid}: {str(e)}")
            
    merge_json_files("/media/desk16/zhiwei/paper_code/MethylFormer/methylformer/data/bin40")


# aws s3 cp train_ctcf_bin40.json s3://demoyc42292/deeplearning/
# aws s3 cp valid_ctcf_bin40.json s3://demoyc42292/deeplearning/
# python scripts/train_ctcf.py  --config configs/train_ctcf_gpu_bin40.json
