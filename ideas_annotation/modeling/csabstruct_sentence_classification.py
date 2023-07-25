import argparse

from datasets import Dataset, load_dataset
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


def csabstruct_to_sentences(dataset):
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
    parser = argparse.ArgumentParser(description="Ideas sentence classification")
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

    dataset = load_dataset("allenai/csabstruct")
    train_dataset = csabstruct_to_sentences(dataset["train"])
    train_tokenized_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "use_context": True},
    )

    validation_dataset = csabstruct_to_sentences(dataset["validation"])
    validation_tokenized_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "use_context": True},
    )

    labels = dataset["train"].info.features["labels"][0].names
    label2id = {k: i for i, k in enumerate(labels)}
    id2label = {i: k for i, k in enumerate(labels)}

    test_dataset = dataset["test"].map(
        lambda example: {
            "labels_str": [id2label[label_id] for label_id in example["labels"]],
            "context": " ".join(example["sentences"]),
        },
        remove_columns=["labels"],
    )
    test_dataset = test_dataset.rename_column("labels_str", "labels")
    test_tokenized_dataset = test_dataset.map(
        preprocess_test_set_function,
        fn_kwargs={"tokenizer": tokenizer, "use_context": True},
    )

    training_args = TrainingArguments(
        output_dir="csabstruct_model",
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
            ["abstract_id", "sentences", "labels", "confs", "context"]
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
