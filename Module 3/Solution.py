
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix



df = pd.read_csv('College.csv')


summary_stats = []
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        mean = df[col].mean()
        std = df[col].std()
        summary_stats.append((col, f"{mean:.2f} ({std:.2f})"))
    elif col != 'Unnamed: 0':
        value_counts = df[col].value_counts()
        for value, count in value_counts.items():
            proportion = count / len(df)
            summary_stats.append((col, f"{value}  {count} ({proportion:.2f})"))

summary_df = pd.DataFrame(summary_stats, columns=['Column Name', 'Descriptive Stats'])
summary_df.to_excel('summary_stats.xlsx', index = False)



df.rename(columns={'Unnamed: 0': 'College Name'}, inplace=True)
df.columns


df['Acceptance Rate'] = (df['Accept'] / df['Apps'] ) * 100
df['Enrollment Rate'] = (df['Enroll'] / df['Accept'] ) * 100


enrollment_rate_df = df[['College Name', 'Acceptance Rate', 'Enrollment Rate']]
enrollment_rate_df['Competitiveness'] = pd.cut(enrollment_rate_df['Acceptance Rate'], bins = 5, right = False, labels = ['Highly Competitive', 'Very Competitive', 'Moderately Competitive', 'Less Competitive', 'Non-Competitive'])
# top_enrollment_rate_df = enrollment_rate_df.sort_values(by='Enrollment Rate', ascending=False).head(10)
fig = px.scatter(
    enrollment_rate_df,
    x = 'Enrollment Rate',
    y = 'Acceptance Rate',
    color = 'Competitiveness'
)
fig.write_image('Enrollment Rate vs Competitiveness.png')
fig.show()


top_10_top_25_df = df[['College Name', 'Acceptance Rate', 'Top10perc', 'Top25perc', 'Enrollment Rate']]
top_10_top_25_df['Perc Top 10 of Top 25'] = top_10_top_25_df['Top10perc'] / top_10_top_25_df['Top25perc']
top_10_top_25_df['Competitiveness'] = pd.cut(top_10_top_25_df['Acceptance Rate'], bins = 5, right = False, labels = ['Highly Competitive', 'Very Competitive', 'Moderately Competitive', 'Less Competitive', 'Non-Competitive'])
top_top_10_top_25_df = top_10_top_25_df.sort_values(by='Perc Top 10 of Top 25', ascending=False).groupby(by='Competitiveness').head(5)
fig = px.scatter(
    top_top_10_top_25_df,
    x = 'Perc Top 10 of Top 25',
    y = 'Enrollment Rate',
    color = 'Competitiveness'
)
fig.write_image('Enrollment Rate vs Percentage of Top 10 of Percentage of Top 25.png')
fig.show()


grad_rate_df = df[['College Name', 'Top10perc', 'Top25perc', 'Acceptance Rate', 'Grad.Rate']]
grad_rate_df['Top 10 of Top 25'] = grad_rate_df['Top10perc'] / grad_rate_df['Top25perc']
grad_rate_df['Competitiveness'] = pd.cut(grad_rate_df['Acceptance Rate'], bins = 5, right = False, labels = ['Highly Competitive', 'Very Competitive', 'Moderately Competitive', 'Less Competitive', 'Non-Competitive'])
top_grad_rate_df = grad_rate_df.sort_values(by='Grad.Rate', ascending=False).groupby(by='Competitiveness').head(5)
fig = px.scatter(
    top_grad_rate_df,
    x = 'Grad.Rate',
    y = 'Top 10 of Top 25',
    color = 'Competitiveness'
)
fig.write_image('Grad Rate vs Percentage of Top 10 of Percentage of Top 25.png')
fig.show()


model_df = df[['College Name', 'Grad.Rate', 'Acceptance Rate', 'Enrollment Rate','Top10perc', 'Top25perc']]
model_df['Top10percOfTop25perc'] = model_df['Top10perc'] / model_df['Top25perc']
model_df.drop(columns=['Top10perc', 'Top25perc'], inplace=True)
model_df['Competitiveness'] = pd.cut(model_df['Acceptance Rate'], bins = 5, right = False, labels = [1, 2, 3, 4, 5])
X = np.asarray(model_df[['Top10percOfTop25perc', 'Grad.Rate', 'Enrollment Rate']])
y = np.asarray(model_df['Competitiveness'].apply(lambda x: 1 if x == 1 else 0))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = sm.add_constant(X_train)


model = sm.GLM(y_train, X_train)
results = model.fit()
results.summary()


X_test = sm.add_constant(X_test)
y_pred = results.predict(X_test)


# Convert probabilities to binary predictions
threshold = 0.5
y_pred_class = (y_pred >= threshold).astype(int)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
specificity = tn / (tn + fp)  # True Negative Rate

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Specificity: {specificity:.2f}")


fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


# Create the ROC curve plot
fig = go.Figure()

# Add ROC curve
fig.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name=f'ROC Curve (AUC = {roc_auc:.2f})',
    line=dict(color='blue')
))

# Add diagonal line for random chance
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Random Chance',
    line=dict(dash='dash', color='gray')
))

# Customize layout
fig.update_layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    xaxis=dict(range=[0.0, 1.0]),
    yaxis=dict(range=[0.0, 1.05]),
    legend=dict(x=0.8, y=0.2),
    template="plotly_white"
)

fig.write_image('ROC Curve.png')
# Show plot
fig.show()


