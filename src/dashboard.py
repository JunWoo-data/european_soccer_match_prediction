# %%
import pandas as pd
import numpy as np 
#import sqlite3 as sql 

import plotly.express as px  
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dash_table, html, dcc

#from flask import Flask
from datetime import date

import joblib

# %%
#server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions=True
server = app.server


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
    "background-color": "#d6e4ea", 
}

CONTENT_STYLE = {
    "margin-left": "16rem",
    "margin-right": "1rem",
    "padding": "1rem 2rem",
    #"background-color": "#edf3f4", 
}



# %%
# ------------------------------------------------------------------------------
# Import and clean data 

# Load basic data
# con = sql.connect("../data/database.sqlite")
# org_league = pd.read_sql(
#     "select * from League", con
#     )

# org_team = pd.read_sql(
#     "select * from Team", con
#     )

org_league = pd.read_csv("../data/org_league.csv")
org_team = pd.read_csv("../data/org_team.csv")

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

# Load data related to modeling
rf_best = joblib.load("../model/best_rf.sav")

df_all = pd.read_csv("../data/df_all.csv")
df_all_std = pd.read_csv("../data/df_all_std.csv")

X_all_train_std = pd.read_csv("../data/X_all_train_std.csv")
X_all_test_std = pd.read_csv("../data/X_all_test_std.csv")

y_all_train_encd = pd.read_csv("../data/y_all_train_encd.csv")["0"]
y_all_test_encd = pd.read_csv("../data/y_all_test_encd.csv")["0"]

# Make match prediction probability data
match_prediction = pd.DataFrame(rf_best.predict_proba(df_all_std.set_index("match_api_id").fillna(0)), 
                                columns = ["away_win_prob", "draw_prob", "home_win_prob"],
                                index = df_all_std.match_api_id).reset_index()

# Load feature importance table
rf_feature_imp_permutation = pd.read_csv("../data/rf_feature_imp_permutation.csv")

# Make team attribute table
home_attribute = df_all[["match_api_id",
                        'home_team_buildUpPlaySpeed',
                        'home_team_buildUpPlayPassing',
                        'home_team_chanceCreationPassing',
                        'home_team_chanceCreationCrossing',
                        'home_team_chanceCreationShooting',
                        'home_team_defencePressure',
                        'home_team_defenceAggression',
                        'home_team_defenceTeamWidth']] \
                .rename(columns = {
                    "home_team_buildUpPlaySpeed": "Build up play speed",
                    "home_team_buildUpPlayPassing": "Build up play passing",
                    "home_team_chanceCreationPassing": "Chance creation passing",
                    "home_team_chanceCreationCrossing": "Chance creation crossing",
                    "home_team_chanceCreationShooting": "Chance creation shooting",
                    "home_team_defencePressure": "Defence pressure",
                    "home_team_defenceAggression": "Defence aggrssion",
                    "home_team_defenceTeamWidth": "Defence team width",
                })

away_attribute = df_all[["match_api_id",
                        'away_team_buildUpPlaySpeed',
                        'away_team_buildUpPlayPassing',
                        'away_team_chanceCreationPassing',
                        'away_team_chanceCreationCrossing',
                        'away_team_chanceCreationShooting',
                        'away_team_defencePressure',
                        'away_team_defenceAggression',
                        'away_team_defenceTeamWidth']] \
                .rename(columns = {
                    "away_team_buildUpPlaySpeed": "Build up play speed",
                    "away_team_buildUpPlayPassing": "Build up play passing",
                    "away_team_chanceCreationPassing": "Chance creation passing",
                    "away_team_chanceCreationCrossing": "Chance creation crossing",
                    "away_team_chanceCreationShooting": "Chance creation shooting",
                    "away_team_defencePressure": "Defence pressure",
                    "away_team_defenceAggression": "Defence aggrssion",
                    "away_team_defenceTeamWidth": "Defence team width",
                })

# Load betting information
df_match_betting = pd.read_csv("../data/df_match_betting.csv")

# Load Elo rating information
elo_rating = pd.read_csv("../data/elo_rating.csv")

# %%                 
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
    style_cell = {'textAlign':'center','minWidth': 40, 'maxWidth': 40, 'width': 40,'font_size': '12px','whiteSpace':'normal','height':'auto'}, 
    # style_cell_conditional = [
    #     {'if': {'column_id': 'Match date'},
    #      'width': '30%'},
    # ]
)

feature_importance_plot = px.box(rf_feature_imp_permutation, x = "Value", y = "feature_set", points="all")
feature_importance_plot.update_layout(
    font = dict(size = 15),
    yaxis_title = None,
    xaxis_title = None,
    margin=dict(l = 10, r = 10, t = 20, b = 20)
)

feature_importance_component = html.Div([
    html.P(" "),
    html.H6("Feature importance from the trained model", style = {'textAlign':'center'}),
    dcc.Graph(id = "feature_importance_plot", figure = feature_importance_plot,
              style = {'height': '400px'})
])

sidebar_component = html.Div(
    [
        html.H4("European Soccer Match Analysis Dashboard", style = {"color": "darkcyan"}),    
        html.Hr(),

        html.P("Select match conditions:"),
        select_league_component,
        html.Br(),
        select_season_component,
        html.Br(),
        select_team_component,
        html.Br(),
        select_date_range_component,
        html.Hr(),
        html.P("Matches with above conditions:"),
        match_table_component
    ], 
    
    style = SIDEBAR_STYLE
)

content_component = html.Div(
    id = "page_content",
    children = [
        dbc.Row([
            dbc.Col(
                html.Div(id = "content_title"),
                width = 6,
                style = {
                    "height": "100px",
                    'backgroundColor': "#edf3f4"
                }
            ),
            
            dbc.Col(
                html.Div(id = "match_prediction_container"),
                width = 5,
                style = {
                    'margin-left': '20px',
                    "height": "100px",
                    'backgroundColor': "#edf3f4"
                }
            ),  
        ]),
        
        dbc.Row([
            dbc.Col(
                html.Div(id = "feature_importance_container"), 
                width = 6,
                style = {
                    'margin-top': '20px',
                    "height": "450px",
                    'backgroundColor': "#edf3f4"
                }
            ),
            
            dbc.Col(
                html.Div(id = "team_attribute_container"), 
                width = 5,
                style = {
                    'margin-top': '20px',
                    'margin-left': '20px',
                    "height": "450px",
                    'backgroundColor': "#edf3f4"
                }
            )
        ]),
        
        dbc.Row([
            dbc.Col(
                html.Div(id = "elo_rate_container"), 
                width = 6,
                style = {
                    'margin-top': '20px',
                    "height": "390px",
                    'backgroundColor': "#edf3f4"
                }
            ),
            
            dbc.Col(
                html.Div(id = "betting_info_container"), 
                width = 5,
                style = {
                    'margin-top': '20px',
                    'margin-left': '20px',
                    "height": "390px",
                    'backgroundColor': "#edf3f4"
                }
            )
        ]),
        
        dbc.Row([html.Div(id = "recent_match_info_container")], 
                 style = {'margin-top': '20px',
                         "height": "400px",
                         'backgroundColor': "#edf3f4"})
        
       
        
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
        
        print(f"Selected match api id: {df[df.index.isin(selected_row)].match_api_id.values[0]}")
    else:
        res = np.nan          
    
    return res

@app.callback(
    [Output(component_id = "content_title", component_property = "children"),
     Output(component_id = "match_prediction_container", component_property = "children"),
     Output(component_id = "feature_importance_container", component_property = "children"),
     Output(component_id = "team_attribute_container", component_property = "children"),
     Output(component_id = "betting_info_container", component_property = "children"),
     Output(component_id = "elo_rate_container", component_property = "children"),
     Output(component_id = "recent_match_info_container", component_property = "children"),],
    Input(component_id = "store_selected_match", component_property = "data")
)
def update_content(data):
    if data: 
        # Load the selected match data
        df = pd.DataFrame(data)
        league = df["League Name"].values[0]
        season = df["season"].values[0]
        stage = df["Stage"].values[0]
        match_date = df["Match date"].values[0]
        home = df.Home_long.values[0]
        away = df.Away_long.values[0]
        match_api_id = df["match_api_id"].values[0]
        home_team_api_id = df.home_team_api_id.values[0]
        away_team_api_id = df.away_team_api_id.values[0]
        
        # content title component
        content_title = html.Div([
            html.P(""),
            html.H4(f"(Home) {home} vs {away} (Away)", style = {'textAlign':'center'}),
            html.P(f"{league} | {season} season | {stage} stage | {match_date}", style = {'textAlign':'center'})
        ])
        
        # match prediction bar chart component
        target_match_prediction = match_prediction[match_prediction.match_api_id == match_api_id]

        match_prediction_chart = px.bar(np.round(target_match_prediction, 2) * 100, 
                                        y = "match_api_id", x = ["home_win_prob", "draw_prob", "away_win_prob"], 
                                        text_auto = True, orientation='h')

        new = {'home_win_prob':'Home win %', 
               'draw_prob': 'Draw %',
               "away_win_prob": "Away win %"}
        
        match_prediction_chart.for_each_trace(lambda t: t.update(name = new[t.name],
                                                    legendgroup = new[t.name]))
        match_prediction_chart.update_traces(textposition = "inside", insidetextanchor = "middle")
        
        match_prediction_chart.update_layout(
            xaxis_title = None,
            yaxis_title = None,
            xaxis = dict(showticklabels = False, ),
            yaxis = dict(showticklabels = False),
            hovermode = False,
            font = dict(size = 15),
            legend = {
                "title": "Match prediction:", 
                "yanchor": "top", 
                "y":0.99,
                "xanchor": "left",
                "x": 0.01,
                "orientation":"h"
            },
            margin=dict(l = 0, r = 0, t = 0, b = 0)
        )
        
        match_prediction_component = html.Div([
            dcc.Graph(id = "match_prediction_chart", figure = match_prediction_chart, 
                      style = {'height': '100px'})
        ])
        
        # team attribute chart component
        target_home_attribute = home_attribute[home_attribute.match_api_id == match_api_id].drop("match_api_id", axis = 1).values[0]
        target_away_attribute = away_attribute[away_attribute.match_api_id == match_api_id].drop("match_api_id", axis = 1).values[0]

        categories = home_attribute.drop("match_api_id", axis = 1).columns

        team_attribute_chart = go.Figure()

        team_attribute_chart.add_trace(go.Scatterpolar(
            r = target_home_attribute,
            theta = categories,
            fill = "toself",
            name = f"{home}"
        ))

        team_attribute_chart.add_trace(go.Scatterpolar(
            r = target_away_attribute,
            theta = categories,
            fill = "toself",
            name = f"{away}"
        ))

        team_attribute_chart.update_layout(
            polar = dict(
                radialaxis = dict(
                    visible = True,
                    range = [0, 100]
                )
            ),
            showlegend = True,
            margin=dict(l = 50, r = 10, t = 20, b = 20)
        )
        
        team_attribute_component = html.Div([
            html.P(""),
            html.H6("Team attributes from FIFA", style = {"textAlign": "center"}),
            dcc.Graph(id = "team_attribute_chart", figure = team_attribute_chart,
                      style = {'height': '400px'})
        ])
        
        # Betting information component
        target_match_bet_info = df_match_betting[df_match_betting.match_api_id == match_api_id]
        
        H_odds = ["B365H", "BWH", "IWH", "LBH", "WHH", "VCH"]
        D_odds = ["B365D", "BWD", "IWD", "LBD", "WHD", "VCD"]
        A_odds = ["B365A", "BWA", "IWA", "LBA", "WHA", "VCA"]
        
        target_match_H_odds = target_match_bet_info[H_odds].melt()
        target_match_D_odds = target_match_bet_info[D_odds].melt()
        target_match_A_odds = target_match_bet_info[A_odds].melt()
        
        betting_info_chart = go.Figure()
        betting_info_chart.add_trace(go.Box(x = target_match_A_odds.value, boxpoints='all', name = "Away win odds"))
        betting_info_chart.add_trace(go.Box(x = target_match_D_odds.value, boxpoints='all', name = "Draw odds"))
        betting_info_chart.add_trace(go.Box(x = target_match_H_odds.value, boxpoints='all', name = "Home win odds"))

        betting_info_chart.update_layout(
            yaxis_title = None,
            font = dict(size = 15),
            legend = {
                        "title": " ", 
                        "yanchor": "top", 
                        "y": 1.1,
                        "xanchor": "left",
                        "x": 0.0,
                        "orientation":"h"
                    },
            margin=dict(l = 10, r = 10, t = 20, b = 20)
        )
        
        betting_info_component = html.Div([
            html.P(""),
            html.H6("Betting odds from 6 differnt websites", style = {"textAlign": "center"}),
            dcc.Graph(id = "betting_info_chart", figure = betting_info_chart,
                      style = {'height': '340px'})
        ])
        
        # Elo rating component
        home_elo_rating = elo_rating[elo_rating.team_api_id == home_team_api_id]
        away_elo_rating = elo_rating[elo_rating.team_api_id == away_team_api_id]
        
        home_elo_rating["date_diff"] = (pd.to_datetime(match_date) - pd.to_datetime(home_elo_rating.elo_target_date)).dt.days
        away_elo_rating["date_diff"] = (pd.to_datetime(match_date) - pd.to_datetime(away_elo_rating.elo_target_date)).dt.days
        
        target_date_home_elo_rating = home_elo_rating[(home_elo_rating.elo_target_date < match_date) & 
                                              (home_elo_rating.date_diff < 365)].sort_values("elo_target_date")

        target_date_away_elo_rating = away_elo_rating[(away_elo_rating.elo_target_date < match_date) & 
                                                      (away_elo_rating.date_diff < 365)].sort_values("elo_target_date")
        
        elo_rate_chart = go.Figure()
        elo_rate_chart.add_trace(go.Scatter(x = target_date_home_elo_rating.elo_target_date, 
                                            y = target_date_home_elo_rating.Elo, 
                                            name = f"{home}"))
        elo_rate_chart.add_trace(go.Scatter(x = target_date_away_elo_rating.elo_target_date, 
                                            y = target_date_away_elo_rating.Elo, 
                                            name = f"{away}"))
        elo_rate_chart.update_layout(
            font = dict(size = 15),
            legend = {
                        "title": " ", 
                        "yanchor": "top", 
                        "y": 1.1,
                        "xanchor": "left",
                        "x": 0.0,
                        "orientation":"h"
                    },
            hovermode = "x unified",
            margin=dict(l = 10, r = 10, t = 20, b = 20)
        )
        
        elo_rate_component = html.Div([
            html.P(""),
            html.H6("Elo rating history", style = {"textAlign": "center"}),
            dcc.Slider(1, 5, None, value = 1, id = "elo_rate_slider",
                        marks={
                            1: '1 year',
                            2: '2',
                            3: '3',
                            4: '4',
                            5: '5'
                        },),
            html.P(""),
            dcc.Graph(id = "elo_rate_chart", figure = elo_rate_chart,
                      style = {"height": "285px"})
        ])
        
        # recent match info component -1.recent win percentage chart
        recent_num_matches = 5
        home_recent_win_pctg = df_all[df_all.match_api_id == match_api_id][[f"home_team_home_win_percentage_last_{recent_num_matches}_matches",
                                                                            f"home_team_home_lose_percentage_last_{recent_num_matches}_matches"]]

        away_recent_win_pctg = df_all[df_all.match_api_id == match_api_id][[f"away_team_away_win_percentage_last_{recent_num_matches}_matches",
                                                                            f"away_team_away_lose_percentage_last_{recent_num_matches}_matches"]]
        
        home_recent_win_pctg[f"home_team_home_draw_percentage_last_{recent_num_matches}_matches"] = 1 - \
        (home_recent_win_pctg[f"home_team_home_win_percentage_last_{recent_num_matches}_matches"] + \
        home_recent_win_pctg[f"home_team_home_lose_percentage_last_{recent_num_matches}_matches"]) 

        away_recent_win_pctg[f"away_team_away_draw_percentage_last_{recent_num_matches}_matches"] = 1 - \
        (away_recent_win_pctg[f"away_team_away_win_percentage_last_{recent_num_matches}_matches"] + \
        away_recent_win_pctg[f"away_team_away_lose_percentage_last_{recent_num_matches}_matches"]) 
        
        home_recent_win_pctg["home_away"] = "home"
        home_recent_win_pctg.rename(
            columns = {
                f"home_team_home_draw_percentage_last_{recent_num_matches}_matches": f"draw_percentage",
                f"home_team_home_win_percentage_last_{recent_num_matches}_matches": f"win_percentage",
                f"home_team_home_lose_percentage_last_{recent_num_matches}_matches": f"lose_percentage",
            }, inplace = True
        )
        
        away_recent_win_pctg["home_away"] = "away"
        away_recent_win_pctg.rename(
            columns = {
                f"away_team_away_draw_percentage_last_{recent_num_matches}_matches": f"draw_percentage",
                f"away_team_away_win_percentage_last_{recent_num_matches}_matches": f"win_percentage",
                f"away_team_away_lose_percentage_last_{recent_num_matches}_matches": f"lose_percentage",
            }, inplace = True
        )
        
        win_pctg = np.round(pd.concat([away_recent_win_pctg, home_recent_win_pctg]), 2)
        
        recent_win_pctg_chart = px.bar(win_pctg, y = "home_away", x = ["win_percentage", "draw_percentage", "lose_percentage"],
                               text_auto = True, color_discrete_sequence = ["#636EFA", "#FECB52", "#EF553B"])

        new = {'win_percentage':'Win %', 
               'draw_percentage': 'Draw %',
               "lose_percentage": "Lose %"}

        recent_win_pctg_chart.for_each_trace(lambda t: t.update(name = new[t.name],
                                             legendgroup = new[t.name]))
        recent_win_pctg_chart.update_traces(textposition = "inside", insidetextanchor = "middle")
        recent_win_pctg_chart.update_layout(
                    xaxis_title = None,
                    yaxis_title = None,
                    xaxis = dict(showticklabels = False,),
                    yaxis = dict(
                        tickvals = ["home", "away"],
                        ticktext = [f"{home}", f"{away}"],
                    ),
                    hovermode = False,
                    font = dict(size = 15),
                    legend = {
                        "title": " ", 
                        "yanchor": "top", 
                        "y":1.1,
                        "xanchor": "left",
                        "x": 0.01,
                        "orientation":"h"
                    },
                    margin = dict(l = 10, r = 10, t = 20, b = 20)
                )
        
        # recent match info component -2.recent goal chart
        home_recent_goals = df_all[df_all.match_api_id == match_api_id][[f"home_team_avg_goal_at_home_last_{recent_num_matches}_matches",
                                                                 f"home_team_avg_oppnt_goal_at_home_last_{recent_num_matches}_matches"]]

        away_recent_goals = df_all[df_all.match_api_id == match_api_id][[f"away_team_avg_goal_at_away_last_{recent_num_matches}_matches",
                                                                         f"away_team_avg_oppnt_goal_at_away_last_{recent_num_matches}_matches"]]
        
        
        home_recent_goals.rename(
            columns = {
                f"home_team_avg_goal_at_home_last_{recent_num_matches}_matches": f"avg_goal",
                f"home_team_avg_oppnt_goal_at_home_last_{recent_num_matches}_matches": f"avg_opponent_goal",
            }, inplace = True
        )
        
        away_recent_goals.rename(
            columns = {
                f"away_team_avg_goal_at_away_last_{recent_num_matches}_matches": f"avg_goal",
                f"away_team_avg_oppnt_goal_at_away_last_{recent_num_matches}_matches": f"avg_opponent_goal",
            }, inplace = True
        )
        
        recent_goal = pd.DataFrame({"value" : [away_recent_goals.avg_goal.values[0],
                                               away_recent_goals.avg_opponent_goal.values[0],
                                               home_recent_goals.avg_goal.values[0],
                                               home_recent_goals.avg_goal.values[0],],
                                    "class": ["avg_goal", "avg_opponent_goal", "avg_goal", "avg_opponent_goal"],
                                    "home_away": ["away", "away", "home", "home"]})
        
        recent_goal = np.round(recent_goal, 2)
        
        recent_goal_fig = px.bar(recent_goal, x = "value", y = "home_away", 
                         color = "class", barmode = 'group', text_auto = True)

        new = {'avg_goal':'Avg goals', 
               'avg_opponent_goal': 'Avg opponent goals'}

        recent_goal_fig.for_each_trace(lambda t: t.update(name = new[t.name],
                                       legendgroup = new[t.name]))
        recent_goal_fig.update_layout(
            xaxis_title = None,
            yaxis_title = None,
            xaxis = dict(showticklabels = False,),
            yaxis = dict(
                tickvals = ["home", "away"],
                ticktext = [f"{home}", f"{away}"],
            ),
            hovermode = False,
            font = dict(size = 15),
            legend = {
                "title": " ", 
                "yanchor": "top", 
                "y":1.1,
                "xanchor": "left",
                "x": 0.01,
                "orientation":"h"
            },
            margin = dict(l = 10, r = 10, t = 20, b = 20)
        )
        
        recent_match_info_component = html.Div([
            html.P(""),
            html.H6("Recent matches information", style = {"textAlign": "center"}),
            dcc.Dropdown(
                options = [
                    {"label": "Recent 1 match", "value": 1},
                    {"label": "Recent 3 matches", "value": 3},
                    {"label": "Recent 5 matches", "value": 5},
                    {"label": "Recent 10 matches", "value": 10},
                    {"label": "Recent 20 matches", "value": 20},
                    {"label": "Recent 30 matches", "value": 30},
                    {"label": "Recent 60 matches", "value": 60},
                    {"label": "Recent 90 matches", "value": 90},
                ],
                multi = False,
                value = 5,
                placeholder = "How many recent matches?",
                style = {'width': "50%"},
                id = "recent_match_info_dropdown"
            ),
            html.P(),
            dbc.Row([
                dbc.Col(dcc.Graph(id = "recent_win_pctg_chart", figure = recent_win_pctg_chart, 
                              style = {"height": "300px"}),
                    width = 6,
                ),
                
                dbc.Col(dcc.Graph(id = "recent_goal_fig", figure = recent_goal_fig, 
                              style = {"height": "300px"}),
                    width = 6,
                )      
            ])
        ])
        
        res = [content_title, match_prediction_component, feature_importance_component, 
               team_attribute_component, betting_info_component, elo_rate_component, recent_match_info_component]
        
    else:
        res = [html.Div([
                    html.P(""),
                    html.H4("Select any match you are interested in at the left table")
                ]),
               html.P(""), html.P(""), html.P(""), html.P(""), html.P(""), html.P("")]
    
    return res

@app.callback(
    [Output(component_id = "recent_win_pctg_chart", component_property = "figure"),
     Output(component_id = "recent_goal_fig", component_property = "figure"),],
    [Input(component_id = "recent_match_info_dropdown", component_property = "value"),
    Input(component_id = "store_selected_match", component_property = "data")]
)
def recent_matches_info_interactive(recent_num_matches, data):
    # Load the selected match data
    df = pd.DataFrame(data)
    match_api_id = df["match_api_id"].values[0]
    home = df.Home_long.values[0]
    away = df.Away_long.values[0]
    
    # recent match info component -1.recent win percentage chart
    home_recent_win_pctg = df_all[df_all.match_api_id == match_api_id][[f"home_team_home_win_percentage_last_{recent_num_matches}_matches",
                                                                        f"home_team_home_lose_percentage_last_{recent_num_matches}_matches"]]
    away_recent_win_pctg = df_all[df_all.match_api_id == match_api_id][[f"away_team_away_win_percentage_last_{recent_num_matches}_matches",
                                                                        f"away_team_away_lose_percentage_last_{recent_num_matches}_matches"]]
    
    home_recent_win_pctg[f"home_team_home_draw_percentage_last_{recent_num_matches}_matches"] = 1 - \
    (home_recent_win_pctg[f"home_team_home_win_percentage_last_{recent_num_matches}_matches"] + \
    home_recent_win_pctg[f"home_team_home_lose_percentage_last_{recent_num_matches}_matches"]) 
    away_recent_win_pctg[f"away_team_away_draw_percentage_last_{recent_num_matches}_matches"] = 1 - \
    (away_recent_win_pctg[f"away_team_away_win_percentage_last_{recent_num_matches}_matches"] + \
    away_recent_win_pctg[f"away_team_away_lose_percentage_last_{recent_num_matches}_matches"]) 
    
    home_recent_win_pctg["home_away"] = "home"
    home_recent_win_pctg.rename(
        columns = {
            f"home_team_home_draw_percentage_last_{recent_num_matches}_matches": f"draw_percentage",
            f"home_team_home_win_percentage_last_{recent_num_matches}_matches": f"win_percentage",
            f"home_team_home_lose_percentage_last_{recent_num_matches}_matches": f"lose_percentage",
        }, inplace = True
    )
    
    away_recent_win_pctg["home_away"] = "away"
    away_recent_win_pctg.rename(
        columns = {
            f"away_team_away_draw_percentage_last_{recent_num_matches}_matches": f"draw_percentage",
            f"away_team_away_win_percentage_last_{recent_num_matches}_matches": f"win_percentage",
            f"away_team_away_lose_percentage_last_{recent_num_matches}_matches": f"lose_percentage",
        }, inplace = True
    )
    
    win_pctg = np.round(pd.concat([away_recent_win_pctg, home_recent_win_pctg]), 2)
    
    recent_win_pctg_chart = px.bar(win_pctg, y = "home_away", x = ["win_percentage", "draw_percentage", "lose_percentage"],
                           text_auto = True, color_discrete_sequence = ["#636EFA", "#FECB52", "#EF553B"])
    
    new = {'win_percentage':'Win %', 
           'draw_percentage': 'Draw %',
           "lose_percentage": "Lose %"}
    
    recent_win_pctg_chart.for_each_trace(lambda t: t.update(name = new[t.name],
                                         legendgroup = new[t.name]))
    recent_win_pctg_chart.update_traces(textposition = "inside", insidetextanchor = "middle")
    recent_win_pctg_chart.update_layout(
                xaxis_title = None,
                yaxis_title = None,
                xaxis = dict(showticklabels = False,),
                yaxis = dict(
                    tickvals = ["home", "away"],
                    ticktext = [f"{home}", f"{away}"],
                ),
                hovermode = False,
                font = dict(size = 15),
                legend = {
                    "title": " ", 
                    "yanchor": "top", 
                    "y":1.1,
                    "xanchor": "left",
                    "x": 0.01,
                    "orientation":"h"
                },
                margin = dict(l = 10, r = 10, t = 20, b = 20)
            )
    
     # recent match info component -2.recent goal chart
    home_recent_goals = df_all[df_all.match_api_id == match_api_id][[f"home_team_avg_goal_at_home_last_{recent_num_matches}_matches",
                                                             f"home_team_avg_oppnt_goal_at_home_last_{recent_num_matches}_matches"]]
    away_recent_goals = df_all[df_all.match_api_id == match_api_id][[f"away_team_avg_goal_at_away_last_{recent_num_matches}_matches",
                                                                     f"away_team_avg_oppnt_goal_at_away_last_{recent_num_matches}_matches"]]
    
    
    home_recent_goals.rename(
        columns = {
            f"home_team_avg_goal_at_home_last_{recent_num_matches}_matches": f"avg_goal",
            f"home_team_avg_oppnt_goal_at_home_last_{recent_num_matches}_matches": f"avg_opponent_goal",
        }, inplace = True
    )
    
    away_recent_goals.rename(
        columns = {
            f"away_team_avg_goal_at_away_last_{recent_num_matches}_matches": f"avg_goal",
            f"away_team_avg_oppnt_goal_at_away_last_{recent_num_matches}_matches": f"avg_opponent_goal",
        }, inplace = True
    )
    
    recent_goal = pd.DataFrame({"value" : [away_recent_goals.avg_goal.values[0],
                                           away_recent_goals.avg_opponent_goal.values[0],
                                           home_recent_goals.avg_goal.values[0],
                                           home_recent_goals.avg_goal.values[0],],
                                "class": ["avg_goal", "avg_opponent_goal", "avg_goal", "avg_opponent_goal"],
                                "home_away": ["away", "away", "home", "home"]})
    
    recent_goal = np.round(recent_goal, 2)
    
    recent_goal_fig = px.bar(recent_goal, x = "value", y = "home_away", 
                     color = "class", barmode = 'group', text_auto = True)
    new = {'avg_goal':'Avg goals', 
           'avg_opponent_goal': 'Avg opponent goals'}
    recent_goal_fig.for_each_trace(lambda t: t.update(name = new[t.name],
                                   legendgroup = new[t.name]))
    recent_goal_fig.update_layout(
        xaxis_title = None,
        yaxis_title = None,
        xaxis = dict(showticklabels = False,),
        yaxis = dict(
            tickvals = ["home", "away"],
            ticktext = [f"{home}", f"{away}"],
        ),
        hovermode = False,
        font = dict(size = 15),
        legend = {
            "title": " ", 
            "yanchor": "top", 
            "y":1.1,
            "xanchor": "left",
            "x": 0.01,
            "orientation":"h"
        },
        margin = dict(l = 10, r = 10, t = 20, b = 20)
    )
    
    return [recent_win_pctg_chart, recent_goal_fig]

@app.callback(
    Output(component_id = "elo_rate_chart", component_property = "figure"),
    [Input(component_id = "elo_rate_slider", component_property = "value"),
    Input(component_id = "store_selected_match", component_property = "data")]
)
def elo_rate_chart_interactive(recent_num_year, data):
    # Load the selected match data
    df = pd.DataFrame(data)
    match_date = df["Match date"].values[0]
    home = df.Home_long.values[0]
    away = df.Away_long.values[0]
    match_api_id = df["match_api_id"].values[0]
    home_team_api_id = df.home_team_api_id.values[0]
    away_team_api_id = df.away_team_api_id.values[0]
    
    # Elo rating chart
    home_elo_rating = elo_rating[elo_rating.team_api_id == home_team_api_id]
    away_elo_rating = elo_rating[elo_rating.team_api_id == away_team_api_id]
    
    home_elo_rating["date_diff"] = (pd.to_datetime(match_date) - pd.to_datetime(home_elo_rating.elo_target_date)).dt.days
    away_elo_rating["date_diff"] = (pd.to_datetime(match_date) - pd.to_datetime(away_elo_rating.elo_target_date)).dt.days
    
    target_date_home_elo_rating = home_elo_rating[(home_elo_rating.elo_target_date < match_date) & 
                                          (home_elo_rating.date_diff < 365 * recent_num_year)].sort_values("elo_target_date")
    target_date_away_elo_rating = away_elo_rating[(away_elo_rating.elo_target_date < match_date) & 
                                                  (away_elo_rating.date_diff < 365 * recent_num_year)].sort_values("elo_target_date")
    
    elo_rate_chart = go.Figure()
    elo_rate_chart.add_trace(go.Scatter(x = target_date_home_elo_rating.elo_target_date, 
                                        y = target_date_home_elo_rating.Elo, 
                                        name = f"{home}"))
    elo_rate_chart.add_trace(go.Scatter(x = target_date_away_elo_rating.elo_target_date, 
                                        y = target_date_away_elo_rating.Elo, 
                                        name = f"{away}"))
    elo_rate_chart.update_layout(
        font = dict(size = 15),
        legend = {
                    "title": " ", 
                    "yanchor": "top", 
                    "y": 1.1,
                    "xanchor": "left",
                    "x": 0.0,
                    "orientation":"h"
                },
        hovermode = "x unified",
        margin=dict(l = 10, r = 10, t = 20, b = 20)
    )
    
    return elo_rate_chart
    
    

# @app.callback(
#     [Output(component_id = "content_title", component_property = "children"),
#      Output(component_id = "match_prediction_container", component_property = "children"),
#      Output(component_id = "feature_importance_container", component_property = "children"),
#      Output(component_id = "team_attribute_container", component_property = "children"),],
#     Input(component_id = "store_selected_match", component_property = "data")
# )
# def update_content(data):
#     if data: 
#         # Load the selected match data
#         df = pd.DataFrame(data)
#         league = df["League Name"].values[0]
#         season = df["season"].values[0]
#         stage = df["Stage"].values[0]
#         match_date = df["Match date"].values[0]
#         home = df.Home_long.values[0]
#         away = df.Away_long.values[0]
#         match_api_id = df["match_api_id"].values[0]
        
#         # content title component
#         content_title = html.Div([
#             html.H4(f"Match analysis for (Home) {home} vs (Away) {away}"),
#             html.P(f"{league} | {season} season | {stage} stage | {match_date}")
#         ])
        
#         # match prediction bar chart component
#         target_match_prediction = match_prediction[match_prediction.match_api_id == match_api_id]

#         match_prediction_chart = px.bar(np.round(target_match_prediction, 2) * 100, 
#                                         y = "match_api_id", x = ["home_win_prob", "draw_prob", "away_win_prob"], 
#                                         text_auto = True, orientation='h')

#         new = {'home_win_prob':'Home win %', 
#                'draw_prob': 'Draw %',
#                "away_win_prob": "Away win %"}
        
#         match_prediction_chart.for_each_trace(lambda t: t.update(name = new[t.name],
#                                                     legendgroup = new[t.name]))
#         match_prediction_chart.update_traces(textposition = "inside", insidetextanchor = "middle")
        
#         match_prediction_chart.update_layout(
#             xaxis_title = None,
#             yaxis_title = None,
#             xaxis = dict(showticklabels = False, ),
#             yaxis = dict(showticklabels = False),
#             hovermode = False,
#             font = dict(size = 15),
#             legend = {
#                 "title": " ", 
#                 "yanchor": "top", 
#                 "y":0.99,
#                 "xanchor": "left",
#                 "x": 0.01,
#                 "orientation":"h"
#             },
#             margin=dict(l = 0, r = 0, t = 0, b = 0)
#         )
        
#         match_prediction_component = html.Div([
#             html.P("Match prediction", style = {'textAlign':'center'}),
#             dcc.Graph(id = "match_prediction_chart", figure = match_prediction_chart)
#         ])
        
#         # team attribute chart component
#         target_home_attribute = home_attribute[home_attribute.match_api_id == match_api_id].drop("match_api_id", axis = 1).values[0]
#         target_away_attribute = away_attribute[away_attribute.match_api_id == match_api_id].drop("match_api_id", axis = 1).values[0]

#         categories = home_attribute.drop("match_api_id", axis = 1).columns

#         team_attribute_chart = go.Figure()

#         team_attribute_chart.add_trace(go.Scatterpolar(
#             r = target_home_attribute,
#             theta = categories,
#             fill = "toself",
#             name = f"{home}"
#         ))

#         team_attribute_chart.add_trace(go.Scatterpolar(
#             r = target_away_attribute,
#             theta = categories,
#             fill = "toself",
#             name = f"{away}"
#         ))

#         team_attribute_chart.update_layout(
#             polar = dict(
#                 radialaxis = dict(
#                     visible = True,
#                     range = [0, 100]
#                 )
#             ),
#             showlegend = True,
#             margin=dict(l = 0, r = 0, t = 0, b = 0)
#         )
        
#         team_attribute_component = html.Div([
#             html.P("Team attributes", style = {"textAlign": "center"}),
#             dcc.Graph(id = "team_attribute_chart", figure = team_attribute_chart)
#         ])

#         res = [content_title, match_prediction_component, feature_importance_component, team_attribute_component]
        
#     else:
#         res = [html.H4("Select any match you are interested in at the left match table"), html.P(""), html.P(""), html.P("")]
    
#     return res

# %%
# ------------------------------------------------------------------------------
# Run the App
if __name__ == "__main__":
    app.run_server(debug = True)