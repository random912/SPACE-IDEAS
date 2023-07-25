import argparse

from datasets import Dataset, load_from_disk
from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)

from ideas_annotation.modeling.idea_dataset_sentence_classification import (
    predict,
    compute_metrics,
    preprocess_function,
    compute_predict_metrics,
    preprocess_test_set_function,
)


def pubmed_to_sentences(dataset):
    sentences = []
    labels = []
    contexts = []
    for example in dataset:
        sentences += example["sentences"]
        labels += example["labels"]
        context = " ".join(example["sentences"])
        contexts += [context] * len(example["labels"])

    ds = Dataset.from_dict(
        {"sentence": sentences, "label": labels, "context": contexts}
    )
    return ds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pubmed 200k RCT sentence classification"
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="data/processed/pubmed-200k-rct",
        help='Path to the training dataset (default "data/processed/pubmed-200k-rct")',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-large",
        help='Huggingface model to finetune (default "roberta-large")',
    )
    args = parser.parse_args()

    return args


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    labels = ["BACKGROUND", "METHODS", "OBJECTIVE", "CONCLUSIONS", "RESULTS"]
    label2id = {k: i for i, k in enumerate(labels)}
    id2label = {i: k for i, k in enumerate(labels)}

    dataset = load_from_disk(args.input_dataset)
    train_dataset = pubmed_to_sentences(dataset["train"])
    train_dataset = train_dataset.map(lambda batch: {"label": label2id[batch["label"]]})
    train_tokenized_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "use_context": True},
    )

    validation_dataset = pubmed_to_sentences(dataset["dev"])
    validation_dataset = validation_dataset.map(
        lambda batch: {"label": label2id[batch["label"]]}
    )
    validation_tokenized_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "use_context": True},
    )

    test_dataset = dataset["test"].map(
        lambda example: {
            "context": " ".join(example["sentences"]),
        }
    )
    test_tokenized_dataset = test_dataset.map(
        preprocess_test_set_function,
        fn_kwargs={"tokenizer": tokenizer, "use_context": True},
    )

    training_args = TrainingArguments(
        output_dir="pubmed_model",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_f1",
        logging_steps=10,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=5, id2label=id2label, label2id=label2id
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=validation_tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    pred = (
        test_tokenized_dataset.remove_columns(
            ["doc_id", "sentences", "labels", "context"]
        )
        .with_format("torch")
        .map(predict, fn_kwargs={"model": model})
    )
    pred_labels = pred["pred_labels"]
    gold_labels = test_tokenized_dataset["labels"]
    metrics = compute_predict_metrics(gold_labels, pred_labels, label2id)
    print(metrics)


if __name__ == "__main__":
    main(parse_args())
