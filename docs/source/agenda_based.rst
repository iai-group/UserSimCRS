Agenda-based simulator
======================

In the agenda-based simulator, the dialogues are simulated by following the principle of Markov Decision Processes. That is, dialogues are seen as sequences of state transitions enabled by intents (or dialogue actions) and states are only dependent on the previous state. 
However, as the agenda-based simulator depends on past observations (i.e., annotated dialogues), it is technically described by a *partially observable markov decision process* (POMDP).

Specifically, the simulator's next action is decided by the agent response.
If the agent responds expectedly, the next action is the top action in the agenda, otherwise the simulator estimates the next action based on the agent intent distribution.

Agenda initialization
---------------------

Before the initialization of the agenda, the simulator needs to know the distribution of intents in the annotated dialogues.
An intent distribution for each participant (i.e., user and agent) is created.
The user intent distribution considers successive user intent pairs, i.e., the intent of the first user utterance and the one from the next user utterance.
The agent intent distribution considers successive agent and user intent pairs, i.e., the intent of the first agent utterance and the intent of the next user utterance.

The agenda is initialized by the following steps:

* Add start intent
* Add next intent based on **user intent distribution** until the next intent is the stopping intent

Agenda update
-------------

The agenda is updated at each user utterance if the agent's intent is expected. In that case, the intent at the top of the agenda is removed.
For example, the initial agenda is ``[start, intent1, intent2, stop]``, after consuming ``start`` and ``intent1`` the agent responds as expected to ``intent1`` then the next user intent is ``intent2`` and the agenda is ``[stop]``.

