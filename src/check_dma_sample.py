import pandas as pd

df = pd.read_csv("data/processed/dma_sample.csv")
known = df[df["Ship type"] != "Undefined"]

per_vessel = known.groupby("MMSI")["Ship type"].nunique()
print("Vessels with 2+ KNOWN ship types:", (per_vessel > 1).sum())
print(known[known["MMSI"].isin(per_vessel[per_vessel > 1].index)].groupby("MMSI")["Ship type"].unique())

print("Vessels with no known ship type at all:", len(set(df["MMSI"]) - set(known["MMSI"])))
print("Vessels labelled both fishing and other:", (df.groupby("MMSI")["label"].nunique() > 1).sum())