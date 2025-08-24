import pandas as pd
import json
result_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/methylformer/data/"
file_id = "ENCFF155PKX"

# methyl_data = pd.read_csv("/media/desk16/zhiwei/paper_code/MethylFormer/processedData/ENCFF004ONU_methyl.csv")
methyl_data = pd.read_csv("/media/desk16/zhiwei/paper_code/MethylFormer/processedData/{}_methyl.csv".format(file_id))


result = methyl_data.iloc[:,:].apply(
    lambda row: {
        "id": row.iloc[0],
        "chr": row.iloc[1].split("_")[0],
        "position": float(row.iloc[1].split("_")[1]),
        "sequence": row.iloc[2:].to_list()
    },
    axis=1
)

with open(result_dir + "train.json", "w", encoding="utf-8") as f:
    json.dump(result.values.tolist(), f, ensure_ascii=False, indent=2)
