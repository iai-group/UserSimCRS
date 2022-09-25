Agenda-based simulator
======================

In the agenda-based simulator, the dialogues are simulated by following the principle of Markov Decision Processes. That is, dialogues are seen as sequences of state transitions enabled by intents (or dialogue actions) and states are only dependent on the previous state. 
However, as the agenda-based simulator depends on past observations (i.e., annotated dialogues), it is technically described by a *partially observable markov decision process* (POMDP).

Specifically, the simulator's next action is decided by the agent response: if the agend responds expectedly, the next action would the the top action in the agenda, otherwise the simulator will repeat itself.

Response generation
-------------------

The response generation module consists of three main steps:

#. Natural language understanding

  * Intent and entity classification

#. Generating response

  * Response intent
  * Response annotations

#. Natural language generation