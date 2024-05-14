# Dialogue files

The YAML files required to train the Rasa DIET classifier can be generated from the annotated dialogues saved in the correct format.
The generation of these files can be done with this command:

```shell
  cd usersimcrs/utils
  python -m annotation_converter_rasa -source PathToAnnotatedDialoguesFile -destination PathToDestinationFolder
```

It will generate the following files:

  - `<originalname>_reformat.yaml`: The original file saved as a yaml file
  - `<originalname>_types_w_examples.yaml`: Slots and example values extracted from the dialogues
  - `<originalname>_rasa_agent.yaml`: Examples of agent utterances for all possible intents/actions that the agent can take
  - `<originalname>_rasa_user.yaml`: Similar to the agent file, but for users
