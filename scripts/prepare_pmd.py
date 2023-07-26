from datasets import Dataset, DatasetDict
import pandas as pd

dataset_dict = {}
for split in ["train", "dev", "test"]:
  data = {}
  with open(f"pubmed-rct/PubMed_20k_RCT/{split}.txt") as f:
    for x in f:
      if x[0]=="#":
        doc_id = x.strip().replace("#","")
        data[doc_id] = {"labels":[],"sentences":[]}
      if x[0].isupper():
        label = x.split("\t")[0]
        sentence = x.split("\t")[1].strip()
        data[doc_id]["labels"].append(label)
        data[doc_id]["sentences"].append(sentence)

  data = [{"doc_id":k,"sentences":v["sentences"],"labels":v["labels"]} for k,v in data.items()]
  dataset_dict[split] = Dataset.from_pandas(pd.DataFrame(data=data))


dataset = DatasetDict(dataset_dict)
dataset.save_to_disk("./data/processed/pubmed-20k-rct")
