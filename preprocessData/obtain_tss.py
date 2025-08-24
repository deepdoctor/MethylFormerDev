import pandas as pd
data_dir = "/media/desk16/share/zhiwei/sequenceData/"
result_dir =  "/media/desk16/zhiwei/paper_code/MethylFormer/result/"
# GTF 文件路径
gtf_file = data_dir + "gencode.v47.annotation.gtf"

# 读取 GTF 文件（跳过注释行）
colnames = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
gtf = pd.read_csv(gtf_file, sep="\t", comment="#", names=colnames)

# 只保留转录本信息
transcripts = gtf[gtf["feature"] == "transcript"].copy()

# 解析 GTF attribute 字段
def parse_attr(attr_str, key):
    for field in attr_str.split(";"):
        field = field.strip()
        if field.startswith(key):
            return field.split('"')[1]
    return None

transcripts["transcript_id"] = transcripts["attribute"].apply(lambda x: parse_attr(x, "transcript_id"))
transcripts["gene_id"] = transcripts["attribute"].apply(lambda x: parse_attr(x, "gene_id"))
transcripts["gene_name"] = transcripts["attribute"].apply(lambda x: parse_attr(x, "gene_name"))

# 计算 TSS
def get_tss(row):
    if row["strand"] == "+":
        return row["start"]
    else:
        return row["end"]

transcripts["TSS"] = transcripts.apply(get_tss, axis=1)

# 选择输出列
tss_df = transcripts[["seqname", "TSS", "strand", "gene_id", "gene_name", "transcript_id"]]

# 保存为 bed 格式 (0-based, TSS ±1 bp 范围)
tss_df_bed = pd.DataFrame({
    "chrom": tss_df["seqname"],
    "start": tss_df["TSS"] - 1,  # BED 格式是 0-based
    "end": tss_df["TSS"],
    "name": tss_df["gene_name"],  # 这里保存基因名
    "transcript_id": tss_df["transcript_id"],
    "score": ".",
    "strand": tss_df["strand"]
})

tss_df_bed.to_csv(result_dir  +"gencode_v47_TSS_with_gene.bed", sep="\t", index=False, header=True)

tss_df_bed_unique = tss_df_bed.drop_duplicates(subset=["name"], keep="first")

tss_df_bed_unique.to_csv(result_dir + "gencode_v47_TSS_with_gene_unique.bed", sep="\t", index=False, header=True)



