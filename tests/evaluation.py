import argparse

import yaml
from dialoguekit.nlu import SatisfactionClassifierSVM
from dialoguekit.utils.dialogue_evaluation import Evaluator
from dialoguekit.utils.dialogue_reader import json_to_dialogues

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dialogues", type=str, help="Path to the dialogues to be evaluated."
)
parser.add_argument("--agent_id", type=str, help="Agent id.")
parser.add_argument("--user_id", type=str, help="User id.")
parser.add_argument(
    "--intent_schema", type=str, help="Path to the intent schema file."
)
args = parser.parse_args()

dialogues = json_to_dialogues(
    args.dialogues, agent_id=args.agent_id, user_id=args.user_id
)
with open(args.intent_schema) as yaml_file:
    reward_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
reward_config = reward_config.get("REWARD")


evaluator = Evaluator(dialogues, reward_config)
avg_turns = evaluator.avg_turns()
reward = evaluator.reward()
satisfaction = evaluator.satisfaction(SatisfactionClassifierSVM())
