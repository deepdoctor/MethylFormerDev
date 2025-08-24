import pandas as pd
import pysam
import numpy as np
from multiprocessing import Pool, cpu_count

# Shared data path and reference genome path
shared_data_dir = "/media/desk16/share/zhiwei/"
fasta_path = shared_data_dir + "sequenceData/hg19.fa"

eid_li = ['E113', 'E112', 'E109', 'E106', 'E105', 'E100', 'E098', 'E097', 'E096', 'E095',
              'E079', 'E066', 'E065', 'E022', 'E021', 'E016', 'E008']
eid = eid_li[1]
# for eid in eid_li:
print(f"Processing {eid} ...")
data_dir = "/media/desk16/share/zhiwei/sequenceData/"
human_TSS = pd.read_csv(data_dir+"human_TSS.bed",sep="\t",header=None)
human_TSS = human_TSS[~human_TSS.iloc[:,0].isin(['chrX', 'chrY', 'chrM'])]
human_TSS = human_TSS.drop_duplicates(subset=[1])

chrom_position = human_TSS.iloc[:,0]+":" + human_TSS.iloc[:,1].astype(str)
import os
os.chdir("/media/desk16/zhiwei/paper_code/CpGPT/")

# Random seed for reproducibility
RANDOM_SEED = 42

# Directory paths
DEPENDENCIES_DIR = "dependencies/"
LLM_DEPENDENCIES_DIR = DEPENDENCIES_DIR + "/human"
DATA_DIR = "data"
PROCESSED_DIR = "data/tutorials/processed/quick_setup"

MODEL_NAME = "cancer"
MODEL_CHECKPOINT_PATH = f"dependencies/model/weights/{MODEL_NAME}.ckpt"
MODEL_CONFIG_PATH = f"dependencies/model/config/{MODEL_NAME}.yaml"
MODEL_VOCAB_PATH = f"dependencies/model/vocab/{MODEL_NAME}.json"

ARROW_DF_PATH = "data/cpgcorpus/raw/GSE182215/GPL13534/betas/QCDPB.arrow"
ARROW_DF_FILTERED_PATH = "data/tutorials/raw/toy_filtered.arrow"

# The maximum context length to give to the model
MAX_INPUT_LENGTH = 20_000 # you might wanna go higher hardware permitting
MAX_ATTN_LENGTH = 1_000
# Standard library imports
import warnings
import os
import json
warnings.simplefilter(action="ignore", category=FutureWarning)

# Plotting imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaging as pya
import seaborn as sns

# Lightning imports
from lightning.pytorch import seed_everything

# cpgpt-specific imports
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.model.cpgpt_module import m_to_beta
# inferencer = CpGPTInferencer(dependencies_dir=DEPENDENCIES_DIR, data_dir=DATA_DIR)
# config = inferencer.load_cpgpt_config(MODEL_CONFIG_PATH)
# # Define datamodule
# quick_setup_datamodule = CpGPTDataModule(
#     predict_dir=PROCESSED_DIR,
#     dependencies_dir=LLM_DEPENDENCIES_DIR,
#     batch_size=1,
#     num_workers=0,
#     max_length=MAX_INPUT_LENGTH,
#     dna_llm=config.data.dna_llm,
#     dna_context_len=config.data.dna_context_len,
#     sorting_strategy=config.data.sorting_strategy,
#     pin_memory=False,
# )

# quick_setup_datamodule.setup()
# first_item = next(iter(quick_setup_datamodule.data_predict))


# Set random seed for reproducibility
seed_everything(RANDOM_SEED, workers=True)

embedder = DNALLMEmbedder(dependencies_dir=LLM_DEPENDENCIES_DIR)

chrom_position = human_TSS.iloc[:,0]+":" + human_TSS.iloc[:,1].astype(str)


chrom_position_sub = [el[3:] for el in chrom_position.values.tolist()]
dna_embedder_result = embedder.parse_dna_embeddings(chrom_position_sub[:100000],species="homo_sapiens")

DNA_embedding_li = []
for i in range(100):
    DNA_embedding = embedder.get_embedding(location = chrom_position_sub[i],species="homo_sapiens",
                               dna_llm = "nucleotide-transformer-v2-500m-multi-species",dna_context_len = 2001)
    DNA_embedding_li.append(DNA_embedding)

print("the DNA embedding is",DNA_embedding_li)
memmap_list = DNA_embedding_li
array_list = [np.array(m) for m in memmap_list]
np.save("/media/desk16/zhiwei/paper_code/CpGPT/dependencies/result/DNA_embedding.npy", array_list, allow_pickle=True)

# 加载
loaded_list = np.load("/media/desk16/zhiwei/paper_code/CpGPT/dependencies/result/DNA_embedding.npy", allow_pickle=True)
