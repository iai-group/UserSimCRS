"""Context model including multiple context dimensions, e.g., temporal."""


from typing import Dict


class ContextModel:
    _DEFAULT_CONTEXT_PROBABILITIES: Dict[str, Dict[str, float]] = dict()

    def __init__(
        self,
        context_probability_mapping: Dict[
            str, Dict[str, float]
        ] = _DEFAULT_CONTEXT_PROBABILITIES,
    ) -> None:
        """Instantiates a context model.

        Args:
            context_probability_mapping: A dictionary with necessary
              probabilities to sample context. If it is not provided, we use
              default values. The dictionary should contain the context
              dimension as the outer key and the corresponding value should be
              another dictionary with key-value pairs for events and the
              respective probability assigned to it. Example structure:
              {
                temporal: {
                    weekend: 0.50,
                    weekday: 0.50
                },
                relational: {
                    group: 0.50,
                    alone: 0.50
                }
              }
        """
        pass

    def sample_context(self):
        """Samples context along each of the dimensions independently.

        Args:

        Returns:
        """
        pass

    def _sample_context_dimension(self, dimension: str):
        """Samples a context along the given dimension.

        Args:
            dimension: The dimension which context is to be sampled along.

        Returns:
        """
        pass
