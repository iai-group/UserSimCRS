Evaluation
==========

The evaluation is performed on the synthetic dialogues generated between the conversational recommender system (CRS) and the user simulator. 
The evaluation scripts are located in the directory `scripts/evaluation`.

Currently, we provide the following evaluation scripts:

  * **Dialogue quality evaluation**: Evaluates the dialogue quality with regards to five aspects: recommendation relevance, communication style, fluency, conversational flow, and overall satisfaction. The scores for each aspect are obtained from a large language model (LLM) hosted on a Ollama server.
  * **Satisfaction evaluation**: Evaluates the user satisfaction using a pre-trained model from DialogueKit.

Please refer to the documentation of each script for more details on how to run them. 