# Daily Sales

Daily Sales is a Streamlit dashboard for exploring daily UPI transaction activity
and estimating daily transaction value from transaction volume and calendar
features.

## Run locally

From this directory, install the dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app uses `UPI_Daily_Statistics_june_2021_to_June_2026.csv` by default.
You can upload a different CSV from the sidebar if it contains `Date`,
`Volume_Million`, and `Value_Crore` columns.

The selected regression model is trained on the earlier dates and evaluated
on a later chronological holdout. The model scores describe that holdout and
are not a guarantee of future prediction accuracy.

## Deploy to Streamlit Community Cloud

1. Push this project, including `streamlit_app.py`, `requirements.txt`, and the
   CSV data file, to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and set
   the main file path to `streamlit_app.py`.
3. Deploy the app.
