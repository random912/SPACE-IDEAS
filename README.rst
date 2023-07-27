===============================
Streamlining Proposal Evaluation in the Open Space Innovation Platform
===============================

|PyPI pyversions|

Source code and datasets for paper Streamlining Proposal Evaluation in the Open Space Innovation Platform using Sequential Sentence Classification and Transfer Learning

Installation
------------

The whole project is handled with ``make``, go to a terminal an issue:

.. code:: bash

   git clone https://github.com/random912/Streamlining-Proposal-Evaluation-in-the-Open-Space-Innovation-Platform.git
   cd Streamlining-Proposal-Evaluation-in-the-Open-Space-Innovation-Platform/
   make setup
   conda activate ideas_annotation
   make install-as-pkg

Reproducibility
---------------
**Data split:**
To split the OSIP dataset in train and test splits, we can run the split_data.py script:

.. code:: bash

   python scripts/split_data.py

Two files, train.jsonl and test.jsonl, will be created in the data/processed folder.

To train a single sentence classifier using the training OSIP data without context, we run:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --input_train_dataset data/processed/train.jsonl --input_test_dataset data/processed/test.jsonl

If we want to use the context, we run:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --input_train_dataset data/processed/train.jsonl --input_test_dataset data/processed/test.jsonl --use_context

To train using the OSIP plus dataset, we have to change the input_train_dataset to :

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --input_train_dataset data/processed/osip_plus.jsonl --input_test_dataset data/processed/test.jsonl --use_context

(TODO: Include how to do the sequential sentence classification)

Sequential Transfer Learning
~~~~~~~~~~~~~~~~~~~~~
We can train a model, using for example osip plus dataset, and use that trained model tu finetune on the OSIP dataset, we can do this with the following command:

.. code:: bash

   python ideas_annotation/modeling/idea_dataset_sentence_classification.py --model $PATH_TO_TRAINED_MODEL --input_train_dataset data/processed/train.jsonl --input_test_dataset data/processed/test.jsonl --use_context

There are different datasets that can be used for this previous fine-tuning, one is OSIP plus, but we can also train other datasets such as: 

- CSAbstruct:

   .. code:: bash
   
      python ideas_annotation/modeling/csabstruct_sentence_classification.py

- PMD20KRCT:

   .. code:: bash

      git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git
      python scripts/prepare_pmd.py
      python ideas_annotation/modeling/pubmed_sentence_classification.py --input_dataset data/processed/pubmed-20k-rct

- Scim:

   .. code:: bash

      python scripts/prepare_scim.py
      python ideas_annotation/modeling/scim_sentence_classification.py --input_dataset data/processed/scim

Multi-Task Learning
~~~~~~~~~~~~~~~~~~~~~

How to cite
-----------

To cite this research please use the following: `TBD`


.. |PyPI pyversions| image:: https://badgen.net/pypi/python/black
   :target: https://www.python.org/
