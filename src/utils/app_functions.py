import os
import pickle
import pandas as pd
import plotly.graph_objects as go

# Function to create a figure for the metrics by threshold
def plot_metrics_by_threshold_app(cv_scores_grouped):
    # Plot the CV scores grouped by the threshold value with plotly graph objects
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cv_scores_grouped.index, y=cv_scores_grouped['test_Accuracy'], name='Accuracy'))
    fig.add_trace(go.Scatter(x=cv_scores_grouped.index, y=cv_scores_grouped['test_Precision'], name='Precision'))
    fig.add_trace(go.Scatter(x=cv_scores_grouped.index, y=cv_scores_grouped['test_Recall'], name='Recall'))
    fig.add_trace(go.Scatter(x=cv_scores_grouped.index, y=cv_scores_grouped['test_F1-score'], name='F1'))
    fig.add_trace(go.Scatter(x=cv_scores_grouped.index, y=cv_scores_grouped['test_AUC'], name='AUC'))
    fig.add_trace(go.Scatter(x=cv_scores_grouped.index, y=cv_scores_grouped['test_Average Precision'], name='AUPRC'))
    fig.add_shape(
        type='line',
        x0=cv_scores_grouped['test_F1-score'].idxmax(),
        y0=0.5,
        x1=cv_scores_grouped['test_F1-score'].idxmax(),
        y1=1,
        line=dict(
            color='black',
            width=1,
            dash='dash'
        ),
        name='Max F1-score'
    )
    
    fig.update_layout(
        height=600,
        width=600,
        xaxis_title='Threshold',
        yaxis_title='Score',
        legend_title='Metric',
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig

# Function to create a figure for the confusion matrix
def plot_confusion_matrix_app(cm, class_names):
    """
    Plot a confusion matrix using Plotly graph objects with actual counts.

    Parameters:
    - cm: Confusion matrix object (2D numpy array) from scikit-learn.
    - class_names: List of class names for labeling the confusion matrix.
    """
    # Create a heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm.astype(str).tolist(),
        texttemplate="%{text}",
        textfont={"size":20},
        colorscale='Viridis',  # You can change the color scale if needed
        colorbar=dict(title='Count'),
    ))

    # Customize the layout
    fig.update_layout(
        xaxis=dict(title='Predicted labels'),
        yaxis=dict(title='True labels'),
        width=600,
        height=600,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig

# Function to load model files from the models folder
def load_model_names(models_folder):
    model_files = os.listdir(models_folder)
    filtered_model_files = [
        model for model in model_files 
        if model.endswith('.pkl') and 
        (not "optuna" in model.replace("_", " ") and not "test" in model.replace("_", " "))
        ]
    return sorted(filtered_model_files)

# Function to load models
def load_model(models_folder, model_name):
    model_file = os.path.join(models_folder, model_name)
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to load cross-validation scores
def load_cv_scores(cv_scores_folder, model_name):
    cv_scores_file = os.path.join(cv_scores_folder, model_name.replace('.pkl', '_cv_scores.csv'))
    df = pd.read_csv(cv_scores_file)
    df_grouped = df.groupby('threshold').mean()
    return df_grouped

# Function to load the shap plots
def get_shap_img_files(shap_img_folder, model_name):

    shap_beeswarm = os.path.join(shap_img_folder, model_name.replace('.pkl', '_bee.png'))
    shap_barplot = os.path.join(shap_img_folder, model_name.replace('.pkl', '_bars.png'))
    
    return [shap_beeswarm, shap_barplot] 