import os
os.getcwd()
os.chdir("/media/desk16/zhiwei/paper_code/CpGPT/")
os.getcwd()

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

# Set random seed for reproducibility
seed_everything(RANDOM_SEED, workers=True)

inferencer = CpGPTInferencer(dependencies_dir=DEPENDENCIES_DIR, data_dir=DATA_DIR)
inferencer.download_dependencies(species="human")

if not os.path.exists(LLM_DEPENDENCIES_DIR):

    # List CpG genomic locations
    example_genomic_locations = ['1:100000', '1:250500', 'X:2031253']

    # Declare required class
    embedder = DNALLMEmbedder(dependencies_dir=LLM_DEPENDENCIES_DIR)

    # Parse the embeddings
    embedder.parse_dna_embeddings(
        example_genomic_locations,
        "homo_sapiens",
        dna_llm="nucleotide-transformer-v2-500m-multi-species",
        dna_context_len=2001,
    )



# Download the checkpoint and configuration files
# inferencer.download_model(MODEL_NAME)
inferencer.download_model(MODEL_NAME)

# Load the model configuration
config = inferencer.load_cpgpt_config(MODEL_CONFIG_PATH)

# quick_setup_datamodule = CpGPTDataModule(
#     predict_dir=PROCESSED_DIR,
#     dependencies_dir=LLM_DEPENDENCIES_DIR,
#     batch_size=1,
#     num_workers=0,
#     max_length=MAX_INPUT_LENGTH,
#     dna_llm=config.data.dna_llm,
#     dna_context_len=config.data.dna_context_len,
#     sorting_strategy=config.data.sorting_ strategy,
#     pin_memory=False,
# )

# Load the model weights
model = inferencer.load_cpgpt_model(
    config,
    model_ckpt_path=MODEL_CHECKPOINT_PATH,
    strict_load=True,
)

# inferencer.download_cpgcorpus_dataset("GSE182215")
inferencer.download_cpgcorpus_dataset("GSE264438")

df = pd.read_feather(ARROW_DF_PATH)
df.set_index('GSM_ID', inplace=True)
df.head()
df.to_feather(ARROW_DF_FILTERED_PATH)

embedder = DNALLMEmbedder(dependencies_dir=LLM_DEPENDENCIES_DIR)
prober = IlluminaMethylationProber(dependencies_dir=LLM_DEPENDENCIES_DIR, embedder=embedder)

quick_setup_datasaver = CpGPTDataSaver(data_paths=ARROW_DF_FILTERED_PATH, processed_dir=PROCESSED_DIR)

# Process the file
quick_setup_datasaver.process_files(prober, embedder)

# Define datamodule
quick_setup_datamodule = CpGPTDataModule(
    predict_dir=PROCESSED_DIR,
    dependencies_dir=LLM_DEPENDENCIES_DIR,
    batch_size=1,
    num_workers=0,
    max_length=MAX_INPUT_LENGTH,
    dna_llm=config.data.dna_llm,
    dna_context_len=config.data.dna_context_len,
    sorting_strategy=config.data.sorting_strategy,
    pin_memory=False,
)

quick_setup_datamodule.setup()
first_item = next(iter(quick_setup_datamodule.data_predict))
# Define datamodule
quick_setup_datamodule_attn = CpGPTDataModule(
    predict_dir=PROCESSED_DIR,
    dependencies_dir=LLM_DEPENDENCIES_DIR,
    batch_size=1,
    num_workers=0,
    max_length=MAX_ATTN_LENGTH,
    dna_llm=config.data.dna_llm,
    dna_context_len=config.data.dna_context_len,
    sorting_strategy=config.data.sorting_strategy,
    pin_memory=False,
)


trainer = CpGPTTrainer(precision="16-mixed")
# inferencer.download_dependencies(species='homo_sapiens', overwrite=True)
quick_setup_sample_embeddings = trainer.predict(
    model=model,
    datamodule=quick_setup_datamodule,
    predict_mode="forward",
    return_keys=["sample_embedding"]
)

array = quick_setup_sample_embeddings["sample_embedding"].cpu().numpy()  # 如果在 GPU 上，要先 tensor.cpu().numpy()

# 保存为 .npy 文件
np.save("/media/desk16/zhiwei/paper_code/CpGPT/dependencies/result/{}_embedding.npy".format("GSE264438"), array)


quick_setup_pred_conditions = trainer.predict(
    model=model,
    datamodule=quick_setup_datamodule,
    predict_mode="forward",
    return_keys=["pred_conditions"]
)

probes = list(df.columns[0:100])
genomic_locations = prober.locate_probes(probes, "homo_sapiens")
quick_setup_pred_meth = trainer.predict(
    model=model,
    datamodule=quick_setup_datamodule,
    predict_mode="reconstruct",
    genomic_locations=genomic_locations,
    species="homo_sapiens",
    return_keys=["pred_meth"],
)

quick_setup_pred_meth["pred_meth"] = m_to_beta(quick_setup_pred_meth["pred_meth"])

quick_setup_pred_meth_cot = trainer.predict(
    model=model,
    datamodule=quick_setup_datamodule,
    predict_mode="reconstruct",
    genomic_locations=genomic_locations,
    species="homo_sapiens",
    n_thinking_steps=5,
    thinking_step_size=1000,
    uncertainty_quantile=0.1,
    return_keys=["pred_meth"],
)

quick_setup_pred_meth_cot["pred_meth"] = m_to_beta(quick_setup_pred_meth_cot["pred_meth"])

quick_setup_attn = trainer.predict(
    model=model,
    datamodule=quick_setup_datamodule_attn,
    predict_mode="attention",
    aggregate_heads="mean",
    layer_index=-1,
    return_keys=["attention_weights", "chroms", "positions", "mask_na", "meth"],
)