Datasets
========

UserSimCRS considers two types of datasets: *dialogues* and *item/rating* datasets. The dialogues are used to train the NLU, NLG, and interaction model components of the simulator. The item and rating datasets are used to create information needs and to model user preferences.

In terms of format, dialogue datasets should be provided in JSON format, where each dialogue follows the DialogueKit schema. Item/rating datasets should be provided in CSV format. For items, the properties considered (columns) are defined in the domain. For ratings, the dataset should include user IDs, item IDs, and rating values (additional columns will be ignored). 

Currently, UserSimCRS only supports the following movie recommendation datasets.


.. list-table:: Dialogue Datasets
   :width: 100%
   :header-rows: 1

   * - Name
     - Description
   * - MovieBot
     - Dialogues between users and `IAI MovieBot v1 <https://github.com/iai-group/MovieBot>`_
   * - `INSPIRED <https://github.com/sweetpeach/Inspired>`_
     - Dialogues obtained via crowdsourcing
   * - `ReDial <https://redialdata.github.io/website/>`_
     - Dialogues obtained via crowdsourcing
   * - `IARD <https://github.com/wanlingcai1997/umap_2020_IARD>`_
     - Subset of ReDial with additional annotations


.. list-table:: Item/Rating Datasets
   :width: 100%
   :header-rows: 1

   * - Name
     - Description
   * - Sample of MovieLens 20M
     - Subset of the `MovieLens 20M dataset <https://grouplens.org/datasets/movielens/20m/>`_ including users and ratings
   * - Sample of MovieLens 25M
     - Subset of the `MovieLens 25M dataset <https://grouplens.org/datasets/movielens/25m/>`_ including users and ratings
   * - INSPIRED Movies
     - Movie database released with the INSPIRED dataset
   * - ReDial
     - Movies and preferences extracted from the ReDial dialogues
