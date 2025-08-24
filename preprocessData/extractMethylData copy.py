import pandas as pd

data_dir = "/media/desk16/share/zhiwei/sequenceData/"
human_TSS = pd.read_csv(data_dir+"human_TSS.bed",sep="\t",header=None)
# human_TSS = pd.read_csv(data_dir+"hg19_TSS.bed",sep="\t",header=None)

human_TSS = human_TSS[human_TSS.iloc[:,1] >= 2100]

# load CTCF and meth data.
import pandas as pd
import pyBigWig
import pysam
import numpy as np
from multiprocessing import Pool, cpu_count

# Shared data path and reference genome path
shared_data_dir = "/media/desk16/share/zhiwei/"
fasta_path = shared_data_dir + "sequenceData/hg38.fa"

# Load CTCF mapping table and build a dictionary from EID to CTCF file ID
eid_ctcf_cell_type = pd.read_csv("/media/desk16/zhiwei/paper_code/methylbert/data/CTCF_data_mapping.csv")
eid_ctcf = dict(zip(eid_ctcf_cell_type["eid"], eid_ctcf_cell_type["ctcf_id"]))

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
import numpy as np

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
    dmr_len = end - start
    
    try:
        # Methylation values
        meth_values = bw_meth.values(chrom, start, end, numpy=True)
        meth_values = np.nan_to_num(meth_values, nan=0.0)
        meth_mean = np.nanmean(meth_values)

        # CTCF values
        ctcf_values = bw_ctcf.values(chrom, start, end, numpy=True)
        ctcf_values = np.nan_to_num(ctcf_values, nan=0.0)

        # DNA sequence
        seq = fasta.fetch(chrom, start, end).upper()

    except Exception as e:
        # Handle errors gracefully
        meth_values = np.array([])
        meth_mean = 0.0
        ctcf_values = np.array([])
        seq = ""
        dmr_len = np.nan
        print(e)
        # optional: print(e) for debugging

    return [chrom, start, end, meth_values, meth_mean, ctcf_values, dmr_len, seq]


if __name__ == "__main__":
    # List of EIDs to process
    eid_li = ['E113', 'E112', 'E109', 'E106', 'E105', 'E100', 'E098', 'E097', 'E096', 'E095',
              'E079', 'E066', 'E065', 'E022', 'E021', 'E016', 'E008']
    eid = eid_li[0]
    human_TSS.iloc[:,1] = human_TSS.iloc[:,1]-2000
    human_TSS.iloc[:,2] = human_TSS.iloc[:,1]+2000
    for eid in eid_li[1:20]:
        print(f"Processing {eid} ...")
        try:
            # File paths for this eid
            bw_meth_path = shared_data_dir + f"roadmap/methylation/{eid}_WGBS_FractionalMethylation.bigwig"
            bw_ctcf_path = shared_data_dir + f"roadmap/CTCF/{eid_ctcf[eid]}.bigWig"
            dmr_path = shared_data_dir + f"roadmap/methylation/{eid}_WGBS_DMRs_v2.bed.gz"

            # Load DMRs for this eid
            dmr_bed = pd.read_csv(dmr_path, sep="\t", header=None)
            dmr_bed.columns = ["chrom", "start", "end", "val1", "val2"]
            regions = dmr_bed[["chrom", "start", "end"]].values.tolist()
            regions = human_TSS.iloc[:,:3].values.tolist()
            # Create multiprocessing pool; each pool processes one EID
            with Pool(processes=cpu_count(), initializer=make_initializer(bw_meth_path, bw_ctcf_path)) as pool:
                result_matrix = pool.map(extract_region_data, regions)
            valid_indices = [idx for idx, item in enumerate(result_matrix) if len(item[3]) > 0]
            valid_gene_name = human_TSS.iloc[:,3].values[valid_indices]
            third_arrays = np.array([item[3] for item in result_matrix if len(item[3]) > 0])
            fivth_arrays = np.array([item[5] for item in result_matrix if len(item[5]) > 0])
            meth_values_df = pd.DataFrame(third_arrays,valid_gene_name).T
            chr_pos = ["chr_"+str(i) for i in range(-1000,1000)]
            meth_values_df.insert(0, "chr_pos", chr_pos)
            meth_values_df.to_csv(f"/media/desk16/zhiwei/paper_code/MethylGPT/tutorials/pretraining/data_example2/{eid}_methyl.csv",index=False)
            # np.save("/media/desk16/zhiwei/paper_code/CpGPT/tutorials/processedData/meth_values.npy", third_arrays)
            # np.save("/media/desk16/zhiwei/paper_code/CpGPT/tutorials/processedData/CTCF_values.npy", fivth_arrays)

            # df_result = pd.DataFrame(result_matrix, columns=["chrom", "start", "end", "meth_values", "meth_mean", "ctcf_values", "dmr_len", "seq"])
            # df_result = df_result[df_result['seq'].str.len() > 0]
            # df_result["eid"] = eid

            # out_path = f"paper_code/CpGPT/tutorials/processedData/{eid}.csv"
            # df_result.to_csv(out_path, index=False)
            # print(f"Finished {eid}, saved to {out_path}, shape: {df_result.shape}")
        except Exception as e:
            print(f"Failed processing {eid}: {str(e)}")
            print(e)


