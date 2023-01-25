Usage
=====

A YAML configuration file is necessary to start the IAI MovieBot; see `default configuration <https://github.com/iai-group/UserSimCRS/blob/main/config/default/config_default.yaml>`_ for an example.  
Run the following command to start the simulation:

.. code-block:: shell

    python -m usersimcrs.run_simulation -c <path_to_config.yaml>


Example
-------

This example shows how to run simulation using the default configuration and the `IAI MovieBot <https://github.com/iai-group/MovieBot>`_ as the conversational agent.

1. Start IAI MovieBot locally

  * Download the IAI MovieBot `here <https://github.com/iai-group/MovieBot/>`_.
  * Checkout to the 'separate-flask-server' branch.
  * Follow the IAI MovieBot installation instructions.
  * Start the IAI MovieBot locally: 
  
.. code-block:: shell
    
    python -m run_bot -c config/moviebot_config_no_integration.yaml`

Note: the parameter `agent_uri` needs to be updated in the configuration in case IAI MovieBot does not run on the default URI (i.e., `http://127.0.0.1:5001`).

2. Run simulation

.. code-block:: shell

    python -m usersimcrs.run_simulation -c config/default/config_default.yaml


After the simulation, the YAML configuration is saved under `data/runs` using the `output_name` parameter.
The simulated dialogue is saved under `dialogue_export`.