# E-commerce Customer Spending Prediction

Streamlit dashboard and reproducible statistical analysis using the UCI Online Retail dataset.

## Dataset
Download the Excel dataset from https://archive.ics.uci.edu/dataset/352/online+retail and save it locally as `data/Online Retail.xlsx`. Do not commit the dataset to a public GitHub repository.

## Install and run
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app also accepts `.xlsx`, `.xls`, or `.csv` uploads through the sidebar.

## Reproduce the analysis
Place the Excel file at `data/Online Retail.xlsx`, then run:
```bash
python analysis.py
```
Outputs are saved to `outputs/`.

## Study design
The default cutoff is 1 September 2011. Customer features are calculated only from transactions before the cutoff; the target is each customer's net revenue on or after the cutoff. This avoids using future-period transaction information as model features. Cancellation invoices and negative-quantity lines are retained in net revenue, while order-behavior features count positive non-cancelled sales. Rows without a customer ID cannot be used for customer-level modelling and are excluded.

## Limitations
The data covers 2010–2011 and may not represent modern e-commerce behavior. Predictions estimate later-period net revenue, not guaranteed future outcomes. Pearson correlation is association, not causation. Results can be sensitive to the cutoff, outliers, returns, and customer activity patterns.

## Attribution
Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Licensed under CC BY 4.0. Verify the dataset's current attribution/license details on the UCI page before final submission.
