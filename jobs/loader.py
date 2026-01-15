import pandas as pd

def load_hdfs_jobs(log_path, label_path, max_jobs=20000):
    labels_df = pd.read_csv(label_path)
    label_map = dict(zip(labels_df["BlockId"], labels_df["Label"]))

    jobs = []

    with open(log_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_jobs:
                break

            if "blk_" not in line:
                continue

            block_id = "blk_" + line.split("blk_")[1].split()[0]
            label = 1 if label_map.get(block_id) == "Anomaly" else 0

            jobs.append({
                "id": len(jobs),
                "text": line.strip(),
                "tokens": len(line.split()),
                "label": label
            })

    print(f"[DATA] Loaded {len(jobs)} jobs")
    return jobs
