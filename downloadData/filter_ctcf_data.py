import pandas as pd
import requests
from tqdm import tqdm  
import os
root_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/downloadData/"
data_dir = "/media/desk16/share/zhiwei/gptGeo/ctcf/"
os.makedirs(data_dir, exist_ok=True)
ctcf_meta = pd.read_csv(root_dir + "ctcf_metadata.tsv",sep="\t")
ctcf_meta["File format"].unique()

ctcf_meta_bigWig = ctcf_meta[ctcf_meta["File format"]=="bigWig"]

for accession, url in zip(ctcf_meta_bigWig["File accession"], ctcf_meta_bigWig["File download URL"]):
    filename = data_dir + f"{accession}.bigWig"   # 以 accession 命名保存
    print(f"downloading {filename} ...")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc=filename, unit="KB"):
                if chunk:
                    f.write(chunk)

        print(f"{filename} download finished")

    except Exception as e:
        print(f"download failed {filename}: {e}")

# nohup python /media/desk16/zhiwei/paper_code/MethylFormer/downloadData/filter_ctcf_data.py > download.log 2>&1 &