import pandas as pd
import requests
import requests
from tqdm import tqdm

root_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/downloadData/"
data_dir = "/media/desk16/share/zhiwei/gptGeo/methyl/"

wgbs_metadata = pd.read_csv(root_dir + "wgbs_metadata.tsv",sep="\t")
wgbs_meta_bigwig = wgbs_metadata[wgbs_metadata["File format"]=="bigWig"]

# 假设 wgbs_meta_bigwig 已经是一个 pandas DataFrame
# 包含 "File accession" 和 "File download URL"
for accession, url in zip(wgbs_meta_bigwig["File accession"], wgbs_meta_bigwig["File download URL"]):
    filename = data_dir + f"{accession}.bigWig"   # 以 accession 命名保存
    print(f"正在下载 {filename} ...")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc=filename, unit="KB"):
                if chunk:
                    f.write(chunk)

        print(f"{filename} 下载完成")

    except Exception as e:
        print(f"下载失败 {filename}: {e}")



