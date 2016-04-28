# CollaborativeRanking

## FILES:
- main.py: the main function
- preprocess.py: contains functions to preprocess the data
- movie_sim.py: contains functions for movie-movie prediction
- user_sim.py: contains functions for user-user prediction
- matrix_factorization.py: contains functions for PMF prediction
- colb_ranking.py: contains functions for collaborative ranking prediction

- `lr` folder: contains the Logistic Regression code from Homework5

## RUNNING COMMAND:
`python main.py [uu|mm|pcc|pmf|colbrk-svm|colbrk-lr] [dot|cosine] [mean|weight] [k] [latent_factor_num] [output_filepath]`
**LR example:** `python main.py colbrk-lr dot mean 5 100 ../../code_output/hw6/lr_100`
