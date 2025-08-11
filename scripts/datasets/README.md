# Add new dataset

This folder contains scripts to add new datasets for the simulators.

For each dataset, a new folder with the formatted dialogues will be created in `data/datasets/`.

## ReDial

To download and format the dialogues to DialogueKit format, use the following command:

```bash
python -m scripts.datasets.redial.redial_to_dialoguekit
```

You can artificially augment the dialogues with an information need and the utterances with dialogue acts using the following command:

```bash
python -m scripts.datasets.redial.augment_redial
```

For more information on the arguments, use the `--help` flag.

## INSPIRED

To download and format the dialogues to DialogueKit format, use the following command:

```bash
python -m scripts.datasets.inspired.inspired_to_dialoguekit
```

Note that ratings cannot be extracted from INSPIRED as we do not have user ids.
