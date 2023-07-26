from datasets import load_dataset, DatasetDict

dataset = load_dataset("scim/naacl_data_prog")
dataset = dataset.map(lambda example: {"sentences":example["sentences"][:10],"labels":example["labels"][:10]})
dataset_split = dataset["train"].train_test_split(0.2)
test_valid = dataset_split["test"].train_test_split(0.5)
dataset = DatasetDict({"train":dataset_split["train"],"validation":test_valid["train"],"test":test_valid["test"]})
dataset.save_to_disk("data/processed/scim")
