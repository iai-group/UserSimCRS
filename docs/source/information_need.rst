Information need
================

The information need defines in a structured manner what the user is looking for. It comprises constraints, requests, and target items. The constraints specify the slot-value pairs that the item of interest must satisfy, while the requests specify the slots for which the user wants
information. The target items represent the "ground truth" items that the user is interested in.

For example, the information need of a user looking for a comedy movie and its associated plot can be represented as follows:

.. code-block:: json

    {
        "constraints": {
            "genre": "comedy"
        },
        "requests": ["plot"],
        "target_items": ["Jump Street", "The Hangover"]
    }

The information need is a core element of the user simulator as it is used to customize the generated responses and can serve for the evaluation of the conversational recommender system.