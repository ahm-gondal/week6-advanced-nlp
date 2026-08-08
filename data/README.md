# Data

  The BBC News dataset (2,225 articles across five categories: business, entertainment, politics, sport, tech) is downloaded automatically by the notebook from a public mirror:

https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv

The CSV is latin-1 encoded and uses the column names 'news' (article text) and 'type' (category). The notebook reads it with encoding='latin-1' and renames these to 'text' and 'category'. No manual download is needed - just run the notebook top to bottom.

  Original source: https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset
