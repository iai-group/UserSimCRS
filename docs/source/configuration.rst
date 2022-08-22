Configuration
=============

The agenda based simulator has a number of parameters that can be customized by
the users of UserSimCRS. These can be either provided in a config file and/or via the command line. Note that arguments passed through the command line will override its counterpart in the config file.

Example
-------

The configuration parameters are expected to be defined in a YAML file.
TODO: Add args for context and persona.

.. code-block:: yaml
  
  ontology: ontology. yaml
  items: movies.csv
  ratings: ratings.csv
  annotated_dialogues: annotated_dialogues.json
  interaction_model: cir6.yaml
  context: 20
  persona: 10

Usage
-----

* All parameters in a YAML file:

.. code-block:: shell

  python -m usersimcrs.run_simulation -config configs/agenda_based.yaml

* Overriding parameters via the command line:

.. code-block:: shell

  python -m usersimcrs.run_simulation -config configs/agenda_based.yaml -ontology data\ontology.yaml
