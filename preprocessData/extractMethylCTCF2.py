import pandas as pd
import pyBigWig
import pysam
import numpy as np
from joblib import Parallel, delayed
import os

shared_data_dir = "/media/desk16/share/zhiwei/"
result_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/result/"
fasta_path = shared_data_dir + "sequenceData/hg38.fa"
saved_data_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/methylformerfull/data/"

human_TSS = pd.read_csv(result_dir + "gencode_v47_TSS_with_gene_unique.bed", sep="\t")
human_TSS = human_TSS[human_TSS.iloc[:, 1] >= 2100]

# 扩展区间上下游
human_TSS.iloc[:, 1] = human_TSS.iloc[:, 1] - 2000
human_TSS.iloc[:, 2] = human_TSS.iloc[:, 2] + 2000

def load_chrom_data(bw_path, chrom):
    """一次性加载整条染色体 bigWig 数据到 numpy 数组"""
    with pyBigWig.open(bw_path) as bw:
        length = bw.chroms(chrom)
        values = bw.values(chrom, 0, length, numpy=True)
        return np.nan_to_num(values, nan=0.0)

def process_region(row, meth_data, ctcf_data, fasta):
    """提取单个区间特征"""
    chrom, start, end = row
    try:
        meth_values = meth_data[chrom][start:end]
        ctcf_values = ctcf_data[chrom][start:end]
        seq = fasta.fetch(chrom, start, end).upper()

        return [chrom, start, end, meth_values.tolist(), ctcf_values.tolist(), seq]

    except Exception as e:
        print(f"Error at {chrom}:{start}-{end} | {e}")
        return [chrom, start, end, [], [], ""]

if __name__ == "__main__":
    eid_li = ["ENCFF990DKJ"]
    eid_ctcf = {"ENCFF990DKJ": "ENCFF334OFY"}

    for eid in eid_li:
        print(f"Processing {eid} ...")
        try:
            bw_meth_path = shared_data_dir + f"gptGeo/ctcfMethyl/{eid}.bigWig"
            bw_ctcf_path = shared_data_dir + f"gptGeo/ctcfMethyl/{eid_ctcf[eid]}.bigWig"

            # 按染色体一次性读取数据
            chroms = human_TSS.iloc[:, 0].unique()
            meth_data = {c: load_chrom_data(bw_meth_path, c) for c in chroms}
            ctcf_data = {c: load_chrom_data(bw_ctcf_path, c) for c in chroms}

            fasta = pysam.FastaFile(fasta_path)
            regions = human_TSS.iloc[:, :3].values.tolist()

            # 并行处理
            result_matrix = Parallel(n_jobs=-1, backend="loky")(
                delayed(process_region)(row, meth_data, ctcf_data, fasta) for row in regions
            )

            # 存成 DataFrame（比 JSON 快很多）
            df = pd.DataFrame(result_matrix,
                              columns=["chr", "position", "end", "methyValue", "ctcfValue", "sequence"])
            out_path = os.path.join(saved_data_dir, "train_ctcf.parquet")
            df.to_parquet(out_path, index=False)
            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"Failed processing {eid}: {str(e)}")
