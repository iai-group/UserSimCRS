# MovieLens-20M sample

This folder contains a small sample from the [MovieLens-20M Dataset](https://grouplens.org/datasets/movielens/20m/).

  * `movies.csv` contains the first 1000 records from the original dataset.
  * `ratings.csv` contains a sample of 1000 ratings for items present in the `movies.csv` file.
    - The file is generated using (1018 is the last movieId in the sample): 
      ```
      ml-20m$ head -n1 ratings.csv >{DIALOGUEKIT}/tests/data/movielens-20m-sample/ratings.csv
      ml-20m$ awk -F, '$2<=1018' ratings.csv '{print }' | head -n 1000 >>{DIALOGUEKIT}/tests/data/movielens-20m-sample/ratings.csv
      ```
