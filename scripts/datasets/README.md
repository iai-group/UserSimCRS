# Add new dataset

This folder contains scripts to add new datasets for the simulators.

## ReDial

To download and format the dialogues to DialogueKit format, use the following command:

```bash
python scripts/datasets/redial/format_redial.py
```

This will create a folder in `data/datasets/` containing the formatted dialogues.

You can artificially augment the dialogues with an information need and the utterances with dialogue acts using the following command:

```bash
python -m scripts.datasets.redial.augment_redial
```

For more information on the arguments, use the `--help` flag.
