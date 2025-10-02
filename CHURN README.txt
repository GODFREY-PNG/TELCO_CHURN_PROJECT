Telco Customer Churn Prediction and Insights Using Machine Learning

Overview

This project leverages machine learning to predict customer churn in a telecommunications company, enabling proactive retention strategies. By analyzing historical customer data, we identify key factors influencing churn and build a robust predictive model using logistic regression. The solution provides actionable insights, such as odds ratios for high-impact features, to guide business decisions like targeted interventions and resource allocation.

Key outcomes include:
- A trained model with high predictive accuracy for churn probability.
- Visualizations highlighting data patterns, correlations, and feature importance.
- Insights into the most influential drivers of churn (e.g., contract type, monthly charges).

This deliverable demonstrates a complete end-to-end pipeline, from data exploration to model deployment, ensuring scalability and interpretability.

Table of Contents
- [Installation]
- [Project Structure
- [Methodology]
- [Key Insights]
- [Model Evaluation]
- [Usage]
- [Contributing]
- [License]
- [Contact]

Installation


1. Set Up Virtual Environment** (Recommended)  
   
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   

2. Install Dependencies  
   Install required libraries  
   
   Core libraries include:
   - pandas` for data manipulation
   - numpy` for numerical operations
   - scikit-learn` for modeling and preprocessing
   - seaborn` and `matplotlib` for visualizations
   - pickle` for model serialization

Project Structure

telco-churn-prediction/
├── data/
│   └── telco_churn.csv          # Raw dataset
├── notebooks/
│   └── 01_churn_analysis.ipynb  # Exploratory Data Analysis (EDA) and Modeling
├── src/
│   ├── data_preprocessing.py    # Data cleaning and feature engineering
│   ├── model_training.py        # Model building and evaluation
│   └── insights.py              # Feature importance and odds ratio calculations
├── models/
│   └── churn_model.pkl          # Saved trained model
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── class_balance.png
│   └── odds_ratio_plot.png
├── requirements.txt              # Dependencies
└── README.md                    # This file

 Methodology

The project follows a structured CRISP-DM (Cross-Industry Standard Process for Data Mining) approach, with a focus on transparency and reproducibility. Below is a high-level overview of the key steps:

1. Data Ingestion and Exploration
- Loaded the Telco customer dataset (`telco_churn.csv`) into a Pandas DataFrame.
- Performed initial investigation using `df.info()` to assess data types, shape, and memory usage.
- Converted object-type columns (e.g., categorical features) to appropriate float types where applicable.

2. Data Cleaning and Preparation
- Identified and handled missing values through imputation or removal.
- Analyzed correlations among numerical features and visualized them via a Seaborn heatmap for quick pattern detection.
- Checked class balance for the target variable (churn: Yes/No) and plotted a bar graph to reveal any imbalances 
- Identified high- and low-cardinality categorical features for optimal encoding strategies.
- Transformed the boolean target to binary (1 for churn, 0 for retention).

3. Feature Engineering and Splitting
- Split data into feature matrix (X) and target vector (y).
- Applied a randomized train-test split (80/20) to ensure unbiased evaluation.

4. Model Building
- Established a baseline model for comparison.
- Built a logistic regression model using a Scikit-learn Pipeline:
  - One-Hot Encoder for categorical variables.
  - Iterative fitting to optimize hyperparameters.
- Generated churn probabilities on the test set for probabilistic predictions.

5. Model Evaluation and Insights
-Evaluated using a Train-Test Accuracy Evaluation to detect potential overfitting.This ensures reliable predictions for high-stakes retention decisions.
- Extracted feature importances and computed odds ratios to quantify impact (EXAMPLE., odds ratio >1 indicates increased churn risk).
- Visualized top odds ratios in a horizontal bar graph, highlighting features like "Month-to-Month Contract" (high positive effect) and "Fiber Optic Internet" (elevated risk).

6. Model Persistence
- Serialized the trained model using Pickle for easy deployment and reloading.

All code is modular, commented, and executable in Jupyter notebooks for interactive exploration.

Key Insights

From the analysis:
- The odds ratios from the logistic regression model in Telco churn analysis reveal how customer features multiplicatively alter churn odds—values above 1 heighten departure risk, below 1 foster retention, and exactly 1 is neutral—visualized in a horizontal bar chart with blue bars extending rightward from feature names, benchmarked by a red dashed line at 1.0 for easy impact assessment. Dominating as the strongest predictors, fiber optic internet (odds ratio ~2.2) likelihood due to unmet premium expectations amid competition, while month-to-month contracts (~2.0) triple it by enabling effortless switches without loyalty locks. Moderate risks emerge from streaming TV and movies (~1.8–2.0), whose commoditized bundles fail to bind users; absent online security or tech support (~1.7 each), which undermine trust in a threat-prone digital world; electronic check payments and paperless billing (~1.6), sparking irritation through transaction glitches or overlooked notices; multiple lines (~1.5), straining networks into frustration; and senior citizen status (~1.4), tied to tech barriers or price sensitivity. Collectively, these highlight churn fueled by service gaps and billing hassles over core flaws, with the model explaining about 25% of variance—pair it with tenure or charges for fuller context. For impact, target fiber/month-to-month users with contract incentives and support boosts, potentially curbing churn 20–30% via proactive perks.

- **Data Quality**: Minimal missing values (handled via mean imputation); moderate class imbalance.

These insights can inform retention campaigns, such as offering discounts to at-risk customers.

[Sample Odds Ratio Plot](visualizations/odds_ratio_plot.png)  
*(Horizontal bar chart showing top 10 features by odds ratio impact on churn.)*

The model outperforms a random baseline  and provides reliable probability scores for risk segmentation.

## Usage

1. **Run EDA and Training**:  
   ```
   jupyter notebook notebooks/01_churn_analysis.ipynb
   ```

2. **Predict on New Data**:  
   ```python
   import pickle
   import pandas as pd

   # Load model
   with open('models/churn_model.pkl', 'rb') as f:
       model = pickle.load(f)

   # Sample new data (DataFrame)
   new_customer = pd.DataFrame({...})  # Your features here
   probabilities = model.predict_proba(new_customer)[:, 1]  # Churn probability
   print(f"Churn Probability: {probabilities[0]:.2%}")
   ```

3. **Generate Insights**:  
   Run `python src/insights.py` to recompute and plot odds ratios.

## Contributing

Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request with detailed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or customizations, reach out to [GODFREY IMBINDI ADEMBESA] at [godfreyimbindi@gmail.com].  

*Project developed with ❤️ using Python and Scikit-learn. Last updated: October 2025.*