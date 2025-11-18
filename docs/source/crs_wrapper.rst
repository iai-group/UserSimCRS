CRS Wrapper
===========

A CRS wrapper facilitates the interaction between a conversational recommender system (CRS) and a simulated user. It provides a standardized communication interface that does not make assumptions about the inner workings or implementation of the CRS. We consider that communication with a CRS is done via an API.
 
In practice, a CRS wrapper is a class inheriting from DialogueKit's `Agent` class with an additional parameter `uri` that specifies the API endpoint of the CRS. It allows to easily support new CRSs by creating new wrappers under `crs_agent_wrapper`.

Supported CRSs
--------------

Below is a list of currently supported CRSs along with their associated wrapper classes and origins.


+-------------------------------------------------------------------------------------------------------------+--------------------------------------------+-----------------------------------------------------------+
| CRS                                                                                                         | Wrapper                                    | Source code                                               |
+=============================================================================================================+============================================+===========================================================+
| IAI MovieBot v1.0.1, `Habib et al., 2020 <https://dl.acm.org/doi/abs/10.1145/3340531.3417433>`_             | sample_agents.moviebot_agent.MovieBotAgent | https://github.com/iai-group/MovieBot/releases/tag/v1.0.1 | 
+-------------------------------------------------------------------------------------------------------------+--------------------------------------------+-----------------------------------------------------------+
| KBRD, `Chen et al., 2019 <https://aclanthology.org/D19-1189/>`_                                             | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS                   |
+-------------------------------------------------------------------------------------------------------------+--------------------------------------------+-----------------------------------------------------------+
| BARCOR, `Wang et al., 2022 <https://arxiv.org/abs/2203.14257>`_                                             | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS                   |
+-------------------------------------------------------------------------------------------------------------+--------------------------------------------+-----------------------------------------------------------+
| UniCRS, `Wang et al., 2022 <https://dl.acm.org/doi/abs/10.1145/3534678.3539382>`_                           | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS                   |
+-------------------------------------------------------------------------------------------------------------+--------------------------------------------+-----------------------------------------------------------+
| ChatCRS, `Wang et al., 2023 <https://aclanthology.org/2023.emnlp-main.621/>`_                               | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS                   |
+-------------------------------------------------------------------------------------------------------------+--------------------------------------------+-----------------------------------------------------------+
| CRB-CRS, `Manzoor and Jannach, 2022 <https://www.sciencedirect.com/science/article/pii/S0306437922000709>`_ | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS                   |
+-------------------------------------------------------------------------------------------------------------+--------------------------------------------+-----------------------------------------------------------+
