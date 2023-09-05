import os
import pickle
import pandas as pd
import plotly.graph_objects as go
from src.utils.app_functions import *
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the path to your models and cv_scores folders
models_folder = 'models'
cv_scores_folder = 'cv_model_scores'
imgs_folder = 'assets'

df = pd.read_csv('data/pulsar_data.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42)

filename_to_string = {
    'rf_baseline.pkl': 'Baseline (No Tuning and No Feat. Eng.)',
    'rf_feat_preproc_tuned_auc.pkl': 'Feature Eng. and Tuned for Max AUC',
    'rf_feat_preproc_tuned_auprc.pkl': 'Feature Eng. and Tuned for Max AUPRC',
    'rf_no_preproc_tuned.pkl': 'Tuned for Max AUPRC (No Feat. Eng.)',
    'rf_feat_preproc_tuned_auprc_dhypm.pkl': 'Feature Eng. and Tuned for Max AUC (Diff. Params. Range)',
}

# Define the app layout
app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(
                html.A(
                    href='https://www.proximus-ada.com/',
                    children=
                        [
                            dbc.Card(
                                dbc.CardImg(
                                    src='assets/ada_symbol.webp'),
                                    className='mb-4 border-0', 
                                    style={'margin':'20px 0px'})
                    ]
                ), 
                width=2
            ),
            dbc.Col(
                html.H1(
                    ["Pulsar Star Classification Results - Random Forest Models"],
                    style={'margin':'20px 0px'}
                ),
                width=10
                )
            ],
        ),
        dbc.Row(html.Hr()),
        dbc.Tabs(
            [dbc.Tab(label=filename_to_string[model], tab_id=model) for model in load_model_names(models_folder)],
            id="tabs", 
            active_tab=load_model_names(models_folder)[0]),
        html.Div(id="content"),
])

# Callback to update the content of each tab
@app.callback(
    Output("content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab_content(active_tab):

    # Load the model and the cross-validation scores
    model = load_model(models_folder, active_tab)
    cv_scores_grouped = load_cv_scores(cv_scores_folder, active_tab)

    # Get the model's parameters
    try:
        # If the model is a pipeline, get the parameters of the rf model
        params = model.get_params()['rf'].get_params()

        # Adjust the names of the parameters, removing the "rf__" prefix
        params = {key.replace('rf__', ''): value for key, value in params.items()}

    except:
        params = model.get_params()

    # Calculate the predicted probabilities and the predicted classes for the optimal threshold
    opt_threshold = cv_scores_grouped['test_F1-score'].idxmax()
    y_pred_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_scores > opt_threshold).astype(int)

    # Create the confusion matrix and the metrics curves by threshold
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = plot_confusion_matrix_app(cm, ['Not Pulsar', 'Pulsar'])
    metrics_fig = plot_metrics_by_threshold_app(cv_scores_grouped)

    # Calculate the metrics
    f1 = round(f1_score(y_test, y_pred), 4)
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    auc = round(roc_auc_score(y_test, y_pred_scores), 4)
    auprc = round(average_precision_score(y_test, y_pred_scores), 4)

    return dbc.Container([
        html.Br(),
        dbc.Row(html.H2(f"Model: {filename_to_string[active_tab]}")),
        html.Br(),
        dbc.Row([dbc.Col(html.H3(f"Metrics Curves by Threshold")), dbc.Col(html.H3("Confusion Matrix"))]),
        dbc.Row([dbc.Col(dcc.Graph(figure=metrics_fig)), dbc.Col(dcc.Graph(figure=cm_fig))]),
        html.Br(),
        dbc.Row(html.H3(f"Best F1 Threshold:")),
        dbc.Row(html.P(f"{round(opt_threshold, 2)}")),
        html.Br(),
        dbc.Row(html.H3("Metrics With Optimal Threshold:")),
        dbc.Row([
            dbc.Col(html.P(f"F1 Score: {f1}"), width=2),
            dbc.Col(html.P(f"Accuracy: {accuracy}"), width=2),
            dbc.Col(html.P(f"Precision: {precision}"), width=2),
            dbc.Col(html.P(f"Recall: {recall}"), width=2),
            dbc.Col(html.P(f"AUC: {auc}"), width=2),
            dbc.Col(html.P(f"AUPRC: {auprc}"), width=2)
        ]),
        html.Br(),
        dbc.Row(html.H3("Model Parameters:")),
        dbc.Row([
            dbc.Col(html.P(f"n_estimators: {params['n_estimators']}"), width=2),
            dbc.Col(html.P(f"min_samples_leaf: {params['min_samples_leaf']}"), width=2),
            dbc.Col(html.P(f"max_features: {params['max_features']}"), width=2),
            dbc.Col(html.P(f"ccp_alpha: {params['ccp_alpha']}"), width=2),
            dbc.Col(html.P(f"class_weight: {params['class_weight']}"), width=2)
        ]),
        html.Br(),
        dbc.Row(html.H3("Model Features Importance (SHAP Values):")),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardImg(src=get_shap_img_files(imgs_folder, active_tab)[0], style={'width':'100%'})), width=6),
            dbc.Col(dbc.Card(dbc.CardImg(src=get_shap_img_files(imgs_folder, active_tab)[1], style={'width':'100%'})), width=6)
        ]),
    ])

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8080, debug=False)
