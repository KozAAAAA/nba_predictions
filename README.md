# NBA Awards Prediction using Random Forest Regressor

The goal of this project is to use machine learning, specifically the Random Forest Regressor, to predict which NBA players will be selected for the All-NBA and All-Rookie teams. These predictions can provide insights for fans, analysts, and teams about player performance and potential future stars.

## Dataset

The dataset used for this project consists of historical NBA player statistics throughout the years acquired using the [NBA API](https://github.com/swar/nba_api). This dataset includes all the necessary per-player performance metrics. Moreover, web scraping techniques were used for year-to-year All-NBA and All-Rookie team selections. Two datasets were then merged by assigning points to players based on their selection to these teams. The All-Rokies and All-NBA datasets are available in the `data/` directory. The final dataset is later divided into training, validation, and test sets.

## Model

To predict the likelihood of a player being selected for the All-NBA or All-Rookie teams, a Random Forest Regressor model is trained using the previously prepared dataset. The model is trained on pairs of player statistics and their corresponding points for being selected to the All-NBA or All-Rookie teams. The model is then used to predict the number of points corresponding to players' likelihood of being selected for these teams. There are two models, one for predicting All-NBA and the other for predicting All-Rookie teams. The top 15 and 10 players, respectively, with the highest predicted number are selected for the teams.

## Usage

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

The dataset preparation, model training, prediction and export of the results are all available in the `nba_predictions.ipynb` notebook.

To run the prediction script, that loads the pre-trained models and predicts the likelihood of players being selected for the All-NBA or All-Rookie teams, use the following command:

```bash
python predict.py <json_path>
```

## Results

The performance of the Random Forest Regressor model was evaluated using a points-based metric, specified in the `nba_predictions.ipynb` notebook. The model was able to predict the All-NBA and All-Rookie teams with a high degree of accuracy. The following are the predicted teams for the 2023-24 season:

| **First All-NBA Team**           | **Second All-NBA Team**       | **Third All-NBA Team**         | **First Rookie All-NBA Team** | **Second Rookie All-NBA Team**      |
|----------------------------------|-------------------------------|--------------------------------|------------------------------|-------------------------------------|
| Luka Doncic                      | LeBron James                  | Domantas Sabonis               | Victor Wembanyama            | Scoot Henderson                      |
| Shai Gilgeous-Alexander          | Kevin Durant                  | DeMar DeRozan                  | Chet Holmgren                | Amen Thompson                        |
| Giannis Antetokounmpo            | Anthony Edwards               | Jalen Brunson                  | Brandon Miller               | Trayce Jackson-Davis                 |
| Nikola Jokic                     | Anthony Davis                 | Pascal Siakam                  | Jaime Jaquez Jr.             | Brandin Podziemski                   |
| Jayson Tatum                     | Kawhi Leonard                 | Tyrese Haliburton              | Keyonte George               | Dereck Lively II                     |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.