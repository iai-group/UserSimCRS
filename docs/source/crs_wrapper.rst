CRS Wrapper
===========

A CRS wrapper facilitates the interaction between a conversational recommender system (CRS) and a simulated user. It provides a standardized communication interface that does not make assumptions about the inner workings or source code of the CRS. We consider that communication with a CRS is done via an API.
 
In practice, a CRS wrapper is a class inheriting from DialogueKit's `Agent` class with an additional parameter `uri` that specifies the API endpoint of the CRS. It allows to easily support new CRSs by creating new wrappers under `crs_agent_wrapper`.

Supported CRSs
--------------

Below is a list of currently supported CRSs along with their associated wrapper class and origins.

| CRS | Wrapper | Source code | Paper |
| --- | ------- | ----------- | ----- |
| IAI MovieBot v1.0.1 | sample_agents.moviebot_agent.MovieBotAgent | https://github.com/iai-group/MovieBot/releases/tag/v1.0.1 | [Habib et al., 2020](https://arxiv.org/abs/2009.03668) |
| KBRD | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS | [Chen et al., 2019](https://arxiv.org/abs/1908.05391) |
| BARCOR | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS | [Wang et al., 2022](https://arxiv.org/abs/2203.14257) |
| UniCRS | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS | [Wang et al., 2022](https://arxiv.org/abs/2206.09363) |
| ChatCRS | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS | [Wang et al., 2023](https://arxiv.org/abs/2305.13112) |
| CRB-CRS | crs_agent_wrapper.ievalm_agent.iEvaLMAgent | https://github.com/iai-group/iEvaLM-CRS | [Manzoor and Jannach, 2022](https://www.sciencedirect.com/science/article/pii/S0306437922000709) |
