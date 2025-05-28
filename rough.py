import chardet

with open("DataCoSupplyChainDataset.csv", "rb") as f:
    result = chardet.detect(f.read(10000))
    print(result)
