Code Layout
===========

This page is a quick map of where the main training and corruption logic lives.
Use it as an index before diving into implementation details.

Core Areas
----------

``nomadflow/data/datasets/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``datamodule.py``: data loading and assembly of train-ready examples.
- Entry point that connects runtime task outputs with featurization.

``nomadflow/data/featurize/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``tokenize.py``: converts structures to token/rigid arrays.
- ``sampling.py``: builds noise/template structures and updates sampled structures.
- ``assembler.py``: turns tokenized data into torch feature tensors and collates batches.
- ``motif_se3_augment.py``: multi-motif rigid-motion augmentation at feature level.

``nomadflow/runtime/training/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Task definitions and per-example conditioning setup.
- ``motif_scaffold.py``: motif-scaffold training task construction.
- ``motif_partition.py``: motif residue partitioning into independent SE(3) motion groups.
- ``partial_seq.py`` and ``unconditional.py``: alternate conditioning modes.

``nomadflow/stoch_interp/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``multiframe.py``: interpolation/corruption process in SE(3) for rigid trajectories.
- Includes dense corruption endpoint sampling and interpolation-time constraints.

Supporting Areas
----------------

``visualize_train_corruption.py`` and ``scripts/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Utilities to inspect corruption trajectories and generate debug visualizations.

``configs/train/``
~~~~~~~~~~~~~~~~~~
- Training/data configuration presets controlling runtime behavior.

``tests/``
~~~~~~~~~~
- Unit/integration checks for featurization, training tasks, and corruption behavior.
- ``test_motif_multi_se3.py`` focuses on multi-motif grouping and motion invariants.
