Setting up an agent
===================


.. contents:: Table of Contents
    :depth: 3

1. Prepare domain and item collection
-------------------------------------

A YAML file with domain-specific slot names must be prepared for the preference model. Additionally, a file containing the item collection is required; currently, this is expected in CSV format.

.. code-block:: YAML
 :caption: domain.yaml

    slot_names:
      TITLE:
      GENRE:
      ACTOR:
      KEYWORD:
      DIRECTOR:

.. code-block:: YAML
 :caption: item_collection.csv

    movieId,title,genres
    1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
    2,Jumanji (1995),Adventure|Children|Fantasy
    3,Grumpier Old Men (1995),Comedy|Romance
    4,Waiting to Exhale (1995),Comedy|Drama|Romance
    5,Father of the Bride Part II (1995),Comedy
    6,Heat (1995),Action|Crime|Thriller
    7,Sabrina (1995),Comedy|Romance


2. Provide preference data
--------------------------

Preference data is consumed in the form of item ratings and needs to be provided in a CSV file in the shape of user ID, item ID, and rating triples.

3. Dialogue sample
------------------

A small sample of dialogues with the target CRS needs to be collected. The sample size depends on the complexity of the system, in terms of action space and language variety, but is generally in the order of 5-50 dialogues


4. Annotate sample 
------------------

The sample of dialogues must contain utterance-level annotations in terms of intents and entities, as this is required to train the NLU and NLG components. Note that the slots used for annotation should be the same as the ones defined in the domain file (cf. Step 1) and intents should follow the ones defined in the interaction model (cf. Step 4.)


5. Define interaction model 
---------------------------

A config file containing the users’ and agent’s intent space, as well as the set of expected agents for each user intent, is required for the interaction model. The CIR6 interaction model shipped with library offers a baseline starting point, which may be further tailored according to the behavior and capabilities of the target CRS

6. Define user model/population
-------------------------------

# TODO Change this when we decide more specifically on our use of context and persona.

Simulation is seeded with a user population that needs to be characterized, in terms of the different contexts (e.g., weekday vs. weekend, alone vs. group setting) and personas (e.g., patient and impatient users) and the number of users to be generated for each combination of context and persona. Each user has their own preference model, which may be instantiated by grounding it to actual preferences (i.e., the ratings dataset given in Step 2)


**Note:** The next three steps are done when running the ``run_simulator.py`` script.

7. Train simulator
------------------

The NLU and NLG components of the simulator are trained using the annotated dialogue sample.

8. Run simulation
-----------------

Running the simulator will generate a set of simulated conversations for each user with the target CRS and save those to files.

9. Perform evaluation
---------------------

Evaluation takes the set of simulated dialogues generated in the previous step as input, and measures the performance of the target CRS in terms of the metrics implemented in DialogueKit (cf. Section 3.1)
