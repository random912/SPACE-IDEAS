===============================
SPACE-IDEAS: A Dataset for Salient Information Detection in Space Innovation
===============================

|PyPI pyversions|

Source code and datasets for paper SPACE-IDEAS: A Dataset for Salient Information Detection in Space Innovation

Installation
------------

The whole project is handled with ``make``, go to a terminal an issue:

.. code:: bash

   git clone https://github.com/random912/SPACE-IDEAS.git
   cd SPACE-IDEAS
   make setup
   conda activate ideas_annotation
   make install-as-pkg

Reproducibility
---------------
**Data split:**
To split the SPACE-IDEAS dataset in train and test splits, we can run the split_data.py script:

.. code:: bash

   python scripts/split_data.py

Two files, train.jsonl and test.jsonl, will be created in the data/processed folder.

**Single-sentence classification:**

To train a single sentence classifier using the training SPACE-IDEAS data without context, we run:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --input_train_dataset data/processed/train.jsonl --input_test_dataset data/processed/test.jsonl

If we want to use the context, we run:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --input_train_dataset data/processed/train.jsonl --input_test_dataset data/processed/test.jsonl --use_context

To train using the SPACE-IDEAS plus dataset, we have to change the input_train_dataset to :

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --input_train_dataset data/processed/osip_plus.jsonl --input_test_dataset data/processed/test.jsonl --use_context

**Sequential sentence classification:**

We need to split the train set in train2 and dev set, we can do this with:

.. code:: bash

   python scripts/split_data.py

Two files, train2.jsonl and dev.jsonl, will be created in the data/processed folder. 

We clone the sequential_sentence_classification repository, create a new conda environment and install the required allennlp library.

.. code:: bash

   git clone https://github.com/random912/sequential_sentence_classification.git
   cd sequential_sentence_classification/
   git checkout allennlp2
   conda create -n sequential_sentence_classification python=3.9
   conda activate sequential_sentence_classification
   pip install allennlp==2.0.0

We have to modify the train.sh script in scripts folder, with the data paths:

.. code:: bash

   TRAIN_PATH=../data/processed/train2.jsonl
   DEV_PATH=../data/processed/dev.jsonl
   TEST_PATH=../data/processed/test.jsonl

We can now run the trainining stript with:

.. code:: bash

   ./scripts/train.sh tmp_output_dir_osip

The trained model will be at tmp_output_dir_osip/model.tar.gz, we can get the test predictions with:

.. code:: bash

   python -m allennlp predict tmp_output_dir_osip/model.tar.gz ../data/processed/test.jsonl --include-package sequential_sentence_classification --predictor SeqClassificationPredictor --cuda-device 0 --output-file osip-predictions.json
   
Now we can obtain the prediction metrics with:

.. code:: bash

   cd ..
   conda activate ideas_annotation
   python scripts/sequential_sentence_classification_metrics.py --prediction_test_file sequential_sentence_classification/osip-predictions.json --gold_test_file data/processed/test.jsonl

Sequential Transfer Learning
~~~~~~~~~~~~~~~~~~~~~
We can train a model, using for example OSIP plus dataset, and use that trained model to finetune on the SPACE-IDEAS dataset, we can do this with the following command:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --model $PATH_TO_TRAINED_MODEL --input_train_dataset data/processed/train.jsonl --input_test_dataset data/processed/test.jsonl --use_context


(TODO: Include how to do it with sequential sentence classification)

Multi-Task Learning
~~~~~~~~~~~~~~~~~~~~~
**Single-sentence classification:**

By deafult, we can do multitask training using all the available datasets (SPACE-IDEAS, SPACE-IDEAS plus) with:

.. code:: bash

   python scripts/merge_osip_dataset.py
   python ideas_annotation/modeling/idea_dataset_multitask_sentence_classification.py

By changing the "tasks" variable in the idea_dataset_multitask_sentence_classification.py script (line 45), we can select the preferred combination of datasets: [ "chatgpt" (SPACE-IDEAS plus), "gold" (SPACE-IDEAS)].

**Sequential sentence classification:**

To run the multitask traininig with sequential sentence classification, we need to install a variation of the `grouphug <https://github.com/sanderland/grouphug>`_ library. We can install it with:

.. code:: bash

   git clone https://github.com/random912/grouphug.git
   cd grouphug
   pip install .
   cd ..

Now we can run the idea_dataset_multitask_sentence_classification.py script:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_multitask_sentence_classification.py

In line 135 of the script, we can set the combinations of datasets that we want to train: ["chatgpt", "gold"].

How to cite
-----------

To cite this research please use the following: `TBD`


.. |PyPI pyversions| image:: https://badgen.net/pypi/python/black
   :target: https://www.python.org/
