# %%
import pandas as pd
import numpy as np 
import sqlite3 as sql 

import plotly.express as px  
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dash_table, html, dcc

from flask import Flask
from datetime import date

import joblib

# %%
server = Flask(__name__)
app = dash.Dash(__name__, server = server, external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

# %%
# ------------------------------------------------------------------------------
# Define default values
DEFAULT_LEAGUE = "Italy Serie A"
DEFAULT_SEASON = "2015/2016"
DEFAULT_START_DATE = date(2016, 5, 8)
DEFAULT_END_DATE = date(2016, 5, 15)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "1rem 1rem",   
    "background-color": "#f8f9fa", 
}

CONTENT_STYLE = {
    "margin-left": "16rem",
    "margin-right": "2rem",
    "padding": "1rem 1rem",
}



# %%
# ------------------------------------------------------------------------------
# Import and clean data 

# Basic data
con = sql.connect("../data/database.sqlite")
org_league = pd.read_sql(
    "select * from League", con
    )

org_team = pd.read_sql(
    "select * from Team", con
    )

df_match_basic = pd.read_csv("../data/df_match_basic.csv")

df_team = pd.read_csv("../data/df_team.csv")

df_match_basic = df_match_basic.merge(org_league.drop("id", axis = 1).rename(columns = {"name": "League Name"}), how = "left", on = "country_id") \
                               .merge(df_team[["team_api_id", "team_long_name"]].drop_duplicates().rename(columns = {"team_long_name": "Home_long", "team_api_id": "home_team_api_id"}), how = "left", on = "home_team_api_id") \
                               .merge(df_team[["team_api_id", "team_long_name"]].drop_duplicates().rename(columns = {"team_long_name": "Away_long", "team_api_id": "away_team_api_id"}), how = "left", on = "away_team_api_id") \
                               .merge(org_team[["team_api_id", "team_short_name"]].drop_duplicates().rename(columns = {"team_short_name": "Home", "team_api_id": "home_team_api_id"}), how = "left", on = "home_team_api_id") \
                               .merge(org_team[["team_api_id", "team_short_name"]].drop_duplicates().rename(columns = {"team_short_name": "Away", "team_api_id": "away_team_api_id"}), how = "left", on = "away_team_api_id") \
                               [["match_api_id", "League Name", "season", "stage", "match_date", "match_result",
                                 "home_team_api_id", "Home_long", "Home", "home_team_goal", 
                                 "away_team_api_id", "Away_long", "Away", "away_team_goal"]] \
                               .rename(columns = {"match_date": "Match date", "stage": "Stage", "home_team_goal": "H goals", "away_team_goal": "A goals"})
df_match_basic.dropna(inplace = True)

# Data related to modeling
rf_best = joblib.load("../model/best_rf.sav")

df_all = pd.read_csv("../data/df_all.csv")
df_all_std = pd.read_csv("../data/df_all_std.csv")

X_all_train_std = pd.read_csv("../data/X_all_train_std.csv")
X_all_test_std = pd.read_csv("../data/X_all_test_std.csv")

y_all_train_encd = pd.read_csv("../data/y_all_train_encd.csv")["0"]
y_all_test_encd = pd.read_csv("../data/y_all_test_encd.csv")["0"]

# Match prediction probability data
match_prediction = pd.DataFrame(rf_best.predict_proba(df_all_std.set_index("match_api_id").fillna(0)), 
                                columns = ["away_win_prob", "draw_prob", "home_win_prob"],
                                index = df_all_std.match_api_id).reset_index()

# %%
# Feature importance table
rf_feature_imp_permutation = pd.read_csv("../data/rf_feature_imp_permutation.csv")

# %%
rf_feature_imp_permutation

                         
# %%
# ------------------------------------------------------------------------------
# Build the Components

select_league_component =  dcc.Dropdown(
    [x for x in sorted(df_match_basic["League Name"].unique())],
    multi = False,
    value = DEFAULT_LEAGUE,
    placeholder = "Select League",
    style = {'width': "100%"},
    id = "select_league"
)

select_season_component = dcc.Dropdown(
    [x for x in sorted(df_match_basic["season"].unique())],
    multi = False,
    value = DEFAULT_SEASON,
    placeholder = "Select Season",
    style = {'width': "100%"},
    id = "select_season"
)

select_date_range_component = dcc.DatePickerRange(
    min_date_allowed = date(2008, 7, 18),
    max_date_allowed = date(2016, 5, 25),
    start_date = DEFAULT_START_DATE,
    end_date = DEFAULT_END_DATE,
    id = "select_date_range"
)

select_team_component = dcc.Dropdown(
    [x for x in pd.concat([df_match_basic["Away_long"], 
                           df_match_basic["Home_long"]]).unique()],
    multi = True,
    placeholder = "Select teams",
    style = {"width": "100%"},
    id = "select_team"
)

match_table_component = dash_table.DataTable(
    data = df_match_basic.to_dict("records"),
    columns = [{"name": i, "id": i} for i in df_match_basic[["Stage", "Home", "H goals", "A goals", "Away"]].columns],
    page_size = 5,
    filter_action = "native",
    id = "match_list",
    row_selectable = "single",
    fill_width = False,
    style_cell={'textAlign':'center','minWidth': 40, 'maxWidth': 40, 'width': 40,'font_size': '12px','whiteSpace':'normal','height':'auto'}, 
)

sidebar_component = html.Div(
    [
        html.H4("European Soccer Match Analysis Dashboard", style = {"color": "darkcyan"}),    
        html.Hr(),

        html.P("Select match conditions: "),
        select_league_component,
        html.Br(),
        select_season_component,
        html.Br(),
        select_team_component,
        html.Br(),
        select_date_range_component,
        html.Hr(),
        html.P("Matches with selected conditions: "),
        match_table_component
    ], 
    
    style = SIDEBAR_STYLE
)

content_component = html.Div(
    id = "page_content",
    children = [
        dbc.Row([
            html.Div(id = "content_title")
        ])
    ],
    style = CONTENT_STYLE
)

# %%
# ------------------------------------------------------------------------------
# Designe the App layout

app.layout = html.Div([
    sidebar_component,
    content_component,
    dcc.Store(id = "store_selected_match")
])

# app.layout = html.Div([
#     dbc.Row(
#         header_component
#     ),
    
#     dbc.Row([
#         dbc.Col(html.P("Select match conditions: ")),
#         dbc.Col(select_league_component),
#         dbc.Col(select_season_component),
#         dbc.Col(select_date_range_component)
#     ]),
    
#     dbc.Row([
#         match_table_component
#     ])
# ])


# %%
# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    Output(component_id = "select_team", component_property = "options"),
    Input(component_id = "select_league", component_property = "value")
)
def make_nested_home_away_dropdown(league):
    dff = df_match_basic.copy()
    
    if league:
        dff = dff[dff["League Name"] == league] 
        
    return [x for x in pd.concat([dff["Away_long"], 
                                  dff["Home_long"]]).unique()]
    
@app.callback(
    Output(component_id = "match_list", component_property = "data"),
    [Input(component_id = "select_league", component_property = "value"),
     Input(component_id = "select_season", component_property = "value"),
     Input(component_id = "select_date_range", component_property = "start_date"),
     Input(component_id = "select_date_range", component_property = "end_date"),
     Input(component_id = "select_team", component_property = "value"),]
)
def update_match_list_table(league, season, start_date, end_date, team):
    print(f"Conditions user chose: {league} | {season} | {start_date} | {end_date} | {team}")    
    
    dff = df_match_basic.copy()
    
    if league:
        dff = dff[dff["League Name"] == league] 
    if season:
        dff = dff[dff["season"] == season]
    if team:
        dff = dff[(dff["Home_long"].isin(team)) | (dff["Away_long"].isin(team))]
    
    dff = dff[pd.to_datetime(dff["Match date"]) >= pd.to_datetime(start_date)]
    dff = dff[pd.to_datetime(dff["Match date"]) <= pd.to_datetime(end_date)]
    
    return dff.to_dict("records")


@app.callback(
    Output(component_id = "store_selected_match", component_property = "data"),
    [Input(component_id = "match_list", component_property = "data"),
     Input(component_id = "match_list", component_property = "selected_rows")]
)
def store_selected_match_to_memory(data, selected_row):
    if selected_row: 
        df = pd.DataFrame(data.copy())
        res = df[df.index.isin(selected_row)].to_dict("records")
        
    else:
        res = np.nan          
    
    return res

@app.callback(
    Output(component_id = "content_title", component_property = "children"),
    Input(component_id = "store_selected_match", component_property = "data")
)
def update_content(data):
    if data: 
        df = pd.DataFrame(data)
        league = df["League Name"].values[0]
        season = df["season"].values[0]
        stage = df["Stage"].values[0]
        match_date = df["Match date"].values[0]
        home = df.Home_long.values[0]
        away = df.Away_long.values[0]
        
        res = html.Div([
            html.H4(f"Match analysis for (Home) {home} vs (Away) {away}"),
            html.P(f"{league} | {season} season | {stage} stage | {match_date}")
        ])
        
    else:
        res = html.H4("Select any match you are interested in at the left match table")
    
    return res

# %%
# ------------------------------------------------------------------------------
# Run the App
app.run_server(debug = True)