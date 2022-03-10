import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


df = pd.read_csv("Training_Set_Values.csv")
df_labels = pd.read_csv("Training_set_labels.csv")

target = pd.read_csv("Training_set_labels.csv")
value_count = target["status_group"].value_counts()
color_list = ["green", "red", "orange"]

dropped_df = df.drop(columns=["longitude", "latitude", "wpt_name", "num_private",
                              "recorded_by", "permit", "payment", "payment_type",
                              "waterpoint_type_group", "basin", "subvillage",
                              "region", "source", "extraction_type_group",
                              "extraction_type_class", "district_code", "quantity_group", "lga", "ward"])
dropped_df["Status"] = df_labels["status_group"]

dropped_df1= dropped_df[dropped_df["population"] != 0]
dropped_df1=pd.get_dummies(dropped_df1, columns=["funder", "installer", "public_meeting", 
                                    "scheme_management", "scheme_name", "extraction_type", 
                                    "management", "management_group", "water_quality", 
                                    "quality_group", "quantity", "source_type", 
                                    "source_class", "waterpoint_type"])

pd.set_option('mode.chained_assignment', None)
dropped_df1["date_recorded"] =pd.to_datetime(dropped_df1["date_recorded"])
dropped_df1.dtypes["date_recorded"]
dropped_df1["year"] = dropped_df1["date_recorded"].dt.year
dropped_df1["month"] = dropped_df1["date_recorded"].dt.month
target= dropped_df1["Status"]
data=dropped_df1.drop(columns=["date_recorded", "id", "Status"])
x_train,x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

KNN_pipe= Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])
KNN_grid= {
    "clf__n_neighbors" : np.arange(5,15,2)
}
KNN_clf = GridSearchCV(KNN_pipe, KNN_grid, n_jobs=3, cv=3, scoring="f1_micro")
print("start knn fit")
KNN_clf.fit(x_train,y_train)
print("knn fit done")

Logres_pipe= Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
Logres_grid= {
    "clf__C" : np.logspace(0.1,1000, num=5)
}
Logres_clf = GridSearchCV(Logres_pipe, Logres_grid, n_jobs=3, cv=3, scoring="f1_micro")
print("logres fit start")
Logres_clf.fit(x_train,y_train)
print("logres fit done")

DTC_pipe= Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier())])
DTC_grid= {
    "clf__max_features" : ["auto", "sqrt", "log2"]
}
DTC_clf = GridSearchCV(DTC_pipe, DTC_grid, n_jobs=3, cv=3, scoring="f1_micro")
print("dtc fit start")
DTC_clf.fit(x_train,y_train)
print("dtc fit done")
model_list= [KNN_clf,Logres_clf,DTC_clf]
model_name_list = ["kNearestNeighbors", "Logistic Regression", "Decision Tree Classifier"]
score_list=[]
label_list=["functional", "functional needs repair","non functional"]

print("starting predictions")
y_pred_knn=KNN_clf.predict(x_test)
y_pred_logres=Logres_clf.predict(x_test)
y_pred_dtc= DTC_clf.predict(x_test)

score_list.append(round(f1_score(y_test,y_pred_knn, average='micro'),4))
score_list.append(round(f1_score(y_test,y_pred_logres, average='micro'),4))
score_list.append(round(f1_score(y_test,y_pred_dtc, average='micro'),4))

print("starting data visualization")
fig1= px.bar(y=model_name_list, x=score_list, hover_name=score_list)
fig1.update_layout(title_text="Classifier Score", title_x=0.5)
fig1.update_xaxes(title_text="Accuracy Score")
fig1.update_yaxes(title_text="Classification Model")
cf1=confusion_matrix(y_test, y_pred_knn, normalize="true", labels=label_list)
cf2=confusion_matrix(y_test, y_pred_logres, normalize="true", labels=label_list)
cf3=confusion_matrix(y_test, y_pred_dtc, normalize="true", labels=label_list)
fig2 =go.Figure()
for cf in [cf1,cf2,cf3]:
    fig2.add_trace(go.Heatmap(
        x=label_list,y=label_list,z=cf, hoverinfo="text", hovertext=np.round(cf,5)
    ))
dropdown_buttons = [  
    {'label': 'KNN', 'method': 'update','args': [{'visible': [True, False, False]}, {'title': 'KNearestNeighbors classification Matrix'}]},  
    {'label': 'LogisticRegression', 'method': 'update','args': [{'visible': [False, True, False]}, {'title': 'LogisticRegression classification Matrix'}]},  
    {'label': "DecisionTreeClassifier", 'method': "update",'args': [{"visible": [False, False, True]}, {'title': 'DecisionTreeClassifier classification Matrix'}]}
    ]

fig2.update_xaxes(title_text="Predicted Label")
fig2.update_yaxes(title_text="True Label")
fig2.update_layout(title_text="KNearestNeighbors classification Matrix")
fig2.data[1].visible=False
fig2.data[2].visible=False
fig2.update_layout(updatemenus=[{'type': "dropdown",'x': 1.25,'y': 0.5,'showactive': True,'active': 0,'buttons': dropdown_buttons}], title_x=0.5)

bar = px.bar(x=value_count.index, y=value_count, color_discrete_sequence=color_list,
             color=value_count.index, title="Target Data Distribution", hover_name=value_count
             )
bar.update_xaxes(title_text="Label")
bar.update_yaxes(title_text="Quantity")

data_cols = [x for x in dropped_df.columns]
d_columns = [{'name': x, 'id': x} for x in data_cols]
d_table = dash_table.DataTable(
    columns=d_columns,
    data=dropped_df.to_dict("records"),
    cell_selectable=False,
    sort_action="native",
    filter_action="native",
    page_action="native",
    page_size=10,
    style_cell={
        'maxwidth':0
    }
)

app = dash.Dash()
app.layout = html.Div([
    html.H1("Predictive Maintenance on Water Pump Dataset"),
    d_table,
    html.P("Dataset Distribution"),
    dcc.Graph(figure=bar),
    html.P("Accuracy Score"),
    dcc.Graph(figure=fig1),
    html.P("Confusion Matrix"),
    dcc.Graph(figure=fig2)],
    style={'text-align': 'center'}
)

if __name__ == '__main__':
    app.run_server(debug=True)
