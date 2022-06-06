"""Generator for different personas with contexts."""

import numpy as np
from typing import Dict, Any, List, Optional
import datetime
import random
import names
import json
from dataclasses import dataclass
from copy import deepcopy

_TIME_OF_THE_DAY_MAPPING = {
    "weekend": {
        (datetime.time(0, 0, 0), datetime.time(3, 0, 0)): {
            "prob": 0.2,
            "patience": 1,
        },
        (datetime.time(3, 0, 0), datetime.time(6, 0, 0)): {
            "prob": 0.075,
            "patience": -2,
        },
        (datetime.time(6, 0, 0), datetime.time(9, 0, 0)): {
            "prob": 0.05,
            "patience": -2,
        },
        (datetime.time(9, 0, 0), datetime.time(12, 0, 0)): {
            "prob": 0.05,
            "patience": 1,
        },
        (datetime.time(12, 0, 0), datetime.time(15, 0, 0)): {
            "prob": 0.075,
            "patience": 1,
        },
        (datetime.time(15, 0, 0), datetime.time(18, 0, 0)): {
            "prob": 0.1,
            "patience": 1,
        },
        (datetime.time(18, 0, 0), datetime.time(21, 0, 0)): {
            "prob": 0.25,
            "patience": 2,
        },
        (datetime.time(21, 0, 0), datetime.time(23, 59, 59)): {
            "prob": 0.2,
            "patience": 2,
        },
    },
    "weekday": {
        (datetime.time(0, 0, 0), datetime.time(3, 0, 0)): {
            "prob": 0.075,
            "patience": -3,
        },
        (datetime.time(3, 0, 0), datetime.time(6, 0, 0)): {
            "prob": 0.05,
            "patience": -2,
        },
        (datetime.time(6, 0, 0), datetime.time(9, 0, 0)): {
            "prob": 0.05,
            "patience": -2,
        },
        (datetime.time(9, 0, 0), datetime.time(12, 0, 0)): {
            "prob": 0.075,
            "patience": -2,
        },
        (datetime.time(12, 0, 0), datetime.time(15, 0, 0)): {
            "prob": 0.1,
            "patience": -1,
        },
        (datetime.time(15, 0, 0), datetime.time(18, 0, 0)): {
            "prob": 0.175,
            "patience": 1,
        },
        (datetime.time(18, 0, 0), datetime.time(21, 0, 0)): {
            "prob": 0.225,
            "patience": 2,
        },
        (datetime.time(21, 0, 0), datetime.time(23, 59, 59)): {
            "prob": 0.250,
            "patience": 1,
        },
    },
}


@dataclass
class Persona:
    """Represents a persons context features."""

    name: str
    id: str
    group_setting: int
    time_of_the_day: datetime
    weekend: bool
    satisfaction: int
    cooperativeness: int


class PersonaGenerator:
    def __init__(self) -> None:
        """Generates personas."""
        self._personas = []
        self._TIME_OF_THE_DAY_MAPPING = _TIME_OF_THE_DAY_MAPPING
        for value in self._TIME_OF_THE_DAY_MAPPING.values():
            summation = 0
            for data in value.values():
                data["cumulated"] = data.get("prob") + summation
                summation += data.get("prob")

    def sample_context(
        self,
        weekend_probability: float = 3 / 7,
        group_probability: float = 3 / 4,
    ) -> Dict[str, Any]:
        """Sample context parameters.

        Samples:
            - group_setting: bool.
            - weekend : bool.
            - time_of_the_day: datetime.
            - cooperativeness: float [0-1].
            - satisfaction: float [0-1] .

        Args:
            weekend_probability: Probability of 'weekend' being 'True'.
            group_probability: Probability of 'group_setting' being 'True'.

        Returns:
            Dictionary containing the mentioned context features.
        """
        # Weekend
        weekend = 1 if random.random() < weekend_probability else 0

        # Group setting
        # scale = 3 if weekend else 0.5
        group_setting = 1 if random.random() < group_probability else 0

        # Time of the day
        uni = random.uniform(0, 1)
        mode = "weekend" if weekend else "weekday"
        for time_range, value in self._TIME_OF_THE_DAY_MAPPING.get(
            mode
        ).items():
            if value.get("cumulated") > uni:
                time_range_selected = time_range
                break

        # Cooperativeness
        cooperativeness = np.random.normal(loc=0.5, scale=0.15)
        cooperativeness = max(min(1, cooperativeness), 0)

        # Satisfaction
        satisfaction = self.satisfaction_function()

        return {
            "weekend": weekend,
            "group_setting": group_setting,
            "time_of_the_day": time_range_selected,
            "cooperativeness": cooperativeness,
            "satisfaction": satisfaction,
        }

    def satisfaction_function(
        self, loc: Optional[float] = 3.0, scale: Optional[float] = 1.0
    ) -> int:
        """Calculates a users satisfaction score.

        Args:
            loc: Mean of the normal distribution.
            scale: Scale of the normal distribution
        Returns:
            Integer satisfaction score [1-5]
        """
        score = round(np.random.normal(loc=loc, scale=scale, size=None))
        score = max(min(5, score), 1)

        return score

    def generate_personas(
        self, amount: Optional[int] = 10, ids: Optional[List[str]] = None
    ) -> List[Persona]:
        """Generates 'amount' of personas.

        Args:
            amount (Optional[int], optional): _description_. Defaults to 10.
            ids (Optional[List[str]], optional): _description_. Defaults to None.

        Returns:
            List[Persona]: _description_
        """
        used_names = set()
        if ids:
            ids = random.sample(ids, amount)
        for _ in range(amount):
            name = names.get_full_name()
            while name in used_names:
                name = names.get_full_name()
            context = self.sample_context(weekend_probability=3 / 7)
            if ids:
                persona_id = ids.pop()
            else:
                persona_id = name
            persona = Persona(
                name=name,
                id=persona_id,
                group_setting=context.get("group_setting"),
                time_of_the_day=context.get("time_of_the_day"),
                weekend=context.get("weekend"),
                satisfaction=context.get("satisfaction"),
                cooperativeness=context.get("cooperativeness"),
            )
            self._personas.append(persona)

        return self._personas

    def export_json(self, filepath: str):
        """Exports the generated personas to filepath as JSON."""
        output_personas = []
        for persona in self._personas:
            p = deepcopy(persona.__dict__)
            p["time_of_the_day"] = (
                p["time_of_the_day"][0].isoformat(),
                p["time_of_the_day"][1].isoformat(),
            )
            output_personas.append(p)
        with open(filepath, "w") as outfile:
            json.dump(output_personas, outfile)
        return output_personas

    def read_json(self, filepath: str) -> List[Persona]:
        """Reads persona JSON from filepath."""
        with open(filepath, "r") as infile:
            data = json.load(infile)
        for person_data in data:
            self._personas.append(
                Persona(
                    name=person_data.get("name"),
                    id=person_data.get("id"),
                    weekend=person_data.get("weekend"),
                    group_setting=person_data.get("group_setting"),
                    time_of_the_day=(
                        datetime.time.fromisoformat(
                            person_data.get("time_of_the_day")[0]
                        ),
                        datetime.time.fromisoformat(
                            person_data.get("time_of_the_day")[1]
                        ),
                    ),
                    cooperativeness=person_data.get("cooperativeness"),
                    satisfaction=person_data.get("satisfaction"),
                )
            )
        return self._personas


if __name__ == "__main__":
    pg = PersonaGenerator()
    persones = pg.generate_personas(amount=1000, ids=list(range(4000, 10000)))
    print(persones)
    pg.export_json("usersimcrs/utils/export_personas.json")
    pg = PersonaGenerator()
    personas = pg.read_json("usersimcrs/utils/export_personas.json")
    print(personas)
