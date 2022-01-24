import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score,cross_validate, GridSearchCV, KFold
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import pickle
from sportsreference.ncaab.teams import Teams
from sportsreference.ncaab.teams import Team
from sportsreference.ncaab.roster import Player
from sklearn.linear_model import LogisticRegression,LinearRegression
import shap
from kenpompy.utils import login
import kenpompy.summary as kp
import kenpompy.misc as kpm

#loading in pickle files:

f = open('kp_login.pickle','rb')
kp_login = pickle.load(f)
f = open('kp_pw.pickle','rb')
kp_pw = pickle.load(f)
f = open('spread_columns.pickle','rb')
spread_cols = pickle.load(f)
f = open('win_loss_columns.pickle','rb')
spread_cols = pickle.load(f)
f = open('wl_model.pickle','rb')
wl_model = pickle.load(f)
f = open('spread_model.pickle','rb')
spread_model = pickle.load(f)
f = open('college_colors.pickle','rb')
college_colors = pickle.load(f)

#Create Kenpom browser 
browser = login(kp_login, kp_pw)

del kp_login, kp_pw

#Establishing list of relevant features for each model:
wl_cols = ['block_percentage',
 'effective_field_goal_percentage',
 'offensive_rating',
 'allowed_effective_field_goal_percentage',
 'allowed_three_point_field_goal_percentage',
 'allowed_total_rebound_percentage',
 'simple_rating_system',
 'three_point_field_goal_percentage',
 'two_point_field_goal_percentage',
 'total_rebound_percentage',
 'turnover_percentage',
 'win_percentage',
 'opp_block_percentage',
 'opp_effective_field_goal_percentage',
 'opp_offensive_rating',
 'opp_simple_rating_system',
 'opp_three_point_field_goal_percentage',
 'opp_two_point_field_goal_percentage',
 'opp_total_rebound_percentage',
 'opp_turnover_percentage',
 'opp_win_percentage',
 'opp_effective_field_goal_percentage_allowed',
 'opp_three_point_field_goal_percentage_allowed',
 'opp_total_rebound_percentage_allowed',
 'home',
 'AdjO',
 'AdjD',
 'Luck',
 'OppO',
 'OppD']

spread_cols = ['simple_rating_system',
 'opp_simple_rating_system',
 'opp_win_percentage',
 'home']

#List to use with pd.map() to quickly rename teams
in_agg = ['Southern Methodist',
          'Brigham Young',
          'University of California',
          'Illinois-Chicago',
          'Connecticut','Massachusetts',
          'Nevada-Las Vegas',
          'North Carolina',
          'North Carolina-Wilmington',
          'North Carolina State',
          'Pittsburgh',
          'Louisiana State',
          'Texas Christian',
          'Southern California',
          'Virginia Commonwealth']

to_change = ['SMU',
             'BYU',
             'California',
             'UIC',
             'UConn',
             'UMass',
             'UNLV',
             'UNC',
             'UNC Wilmington',
             'NC State',
             'Pitt',
             'LSU',
             'TCU',
             'USC',
             'VCU']

missing = ['Loyola (IL)',
     'Louisiana-Monroe',
     'UNC',
     'Miami (FL)',
     'UConn',
     'College of Charleston',
     'Alabama-Birmingham',
     'Albany (NY)',
     'St. Francis (NY)',
     'Prairie View',
     'Savannah St.',
     'Omaha',
     'Maryland-Eastern Shore',
     'Texas-Arlington',
     'Loyola (MD)',
     'NC St.',
     'Pitt',
     'Citadel',
     'Grambling',
     "St. John's (NY)",
     'UC-Irvine',
     'UIC',
     'California Baptist',
     'Bowling Green St.',
     'Bethune-Cookman',
     'Gardner-Webb',
     'Florida International',
     'Texas-Rio Grande Valley',
     'Saint Francis (PA)',
     'UC-Riverside',
     'Purdue-Fort Wayne',
     'UMass',
     'UC-Davis',
     'Miami (OH)',
     'Texas A&M-Corpus Christi',
     'Arkansas-Pine Bluff',
     'UC-San Diego']

in_kp = ['Loyola Chicago',
         'Louisiana Monroe',
         'North Carolina',
         'Miami FL',
         'Connecticut',
         'Charleston',
         'UAB',
         'Albany',
         'St. Francis NY',
         'Prairie View A&M',
         'Savannah St.',
         'Nebraska Omaha',
         'Maryland Eastern Shore',
         'UT Arlington',
         'Loyola MD',
         'N.C. State',
         'Pittsburgh',
         'The Citadel',
         'Grambling St.',
         "St. John's",
         'UC Irvine',
         'Cal Baptist',
        'Bowling Green',
         'Bethune Cookman',
         'Gardner Webb',
         'FIU',
         'UT Rio Grande Valley',
         'St. Francis PA',
        'UC Riverside',
         'Purdue Fort Wayne',
         'Massachusetts',
         'UC Davis',
         'Miami OH',
         'Texas A&M Corpus Christi',
          'Arkansas Pine Bluff',
         'UC San Diego']

#A list of columns I want when creating the KenPom dataframe
kp_cols = ['Rk', 'Team', 'Conf', 'W-L', 'AdjEM', 'AdjO', 'AdjO_r', 'AdjD', 'AdjD_R','AdjT', 'AdjT_r', 'Luck', 'Luck_r', 
         'AdjEM', 'AdjEM_r', 'OppO', 'OppO_r','OppD', 'OppD_r', 'AdjEM', 'AdjEM_r', 'Seed']

#Just a list of valid team names
valid_team_names = ['Abilene Christian',
 'Air Force',
 'Akron',
 'Alabama A&M',
 'Alabama-Birmingham',
 'Alabama State',
 'Alabama',
 'Albany (NY)',
 'Alcorn State',
 'American',
 'Appalachian State',
 'Arizona State',
 'Arizona',
 'Little Rock',
 'Arkansas-Pine Bluff',
 'Arkansas State',
 'Arkansas',
 'Army',
 'Auburn',
 'Austin Peay',
 'Ball State',
 'Baylor',
 'Bellarmine',
 'Belmont',
 'Bethune-Cookman',
 'Binghamton',
 'Boise State',
 'Boston College',
 'Boston University',
 'Bowling Green State',
 'Bradley',
 'BYU',
 'Brown',
 'Bryant',
 'Bucknell',
 'Buffalo',
 'Butler',
 'Cal Poly',
 'Cal State Bakersfield',
 'Cal State Fullerton',
 'Cal State Northridge',
 'California Baptist',
 'UC-Davis',
 'UC-Irvine',
 'UC-Riverside',
 'UC-San Diego',
 'UC-Santa Barbara',
 'California',
 'Campbell',
 'Canisius',
 'Central Arkansas',
 'Central Connecticut State',
 'Central Florida',
 'Central Michigan',
 'Charleston Southern',
 'Charlotte',
 'Chattanooga',
 'Chicago State',
 'Cincinnati',
 'The Citadel',
 'Clemson',
 'Cleveland State',
 'Coastal Carolina',
 'Colgate',
 'College of Charleston',
 'Colorado State',
 'Colorado',
 'Columbia',
 'UConn',
 'Coppin State',
 'Cornell',
 'Creighton',
 'Dartmouth',
 'Davidson',
 'Dayton',
 'Delaware State',
 'Delaware',
 'Denver',
 'DePaul',
 'Detroit Mercy',
 'Dixie State',
 'Drake',
 'Drexel',
 'Duke',
 'Duquesne',
 'East Carolina',
 'East Tennessee State',
 'Eastern Illinois',
 'Eastern Kentucky',
 'Eastern Michigan',
 'Eastern Washington',
 'Elon',
 'Evansville',
 'Fairfield',
 'Fairleigh Dickinson',
 'Florida A&M',
 'Florida Atlantic',
 'Florida Gulf Coast',
 'Florida International',
 'Florida State',
 'Florida',
 'Fordham',
 'Fresno State',
 'Furman',
 'Gardner-Webb',
 'George Mason',
 'George Washington',
 'Georgetown',
 'Georgia Southern',
 'Georgia State',
 'Georgia Tech',
 'Georgia',
 'Gonzaga',
 'Grambling',
 'Grand Canyon',
 'Green Bay',
 'Hampton',
 'Hartford',
 'Harvard',
 'Hawaii',
 'High Point',
 'Hofstra',
 'Holy Cross',
 'Houston Baptist',
 'Houston',
 'Howard',
 'Idaho State',
 'Idaho',
 'UIC',
 'Illinois State',
 'Illinois',
 'Incarnate Word',
 'Indiana State',
 'Indiana',
 'Iona',
 'Iowa State',
 'Iowa',
 'Purdue-Fort Wayne',
 'IUPUI',
 'Jackson State',
 'Jacksonville State',
 'Jacksonville',
 'James Madison',
 'Kansas State',
 'Kansas',
 'Kennesaw State',
 'Kent State',
 'Kentucky',
 'La Salle',
 'Lafayette',
 'Lamar',
 'Lehigh',
 'Liberty',
 'Lipscomb',
 'Cal State Long Beach',
 'Long Island University',
 'Longwood',
 'Louisiana',
 'Louisiana-Monroe',
 'LSU',
 'Louisiana Tech',
 'Louisville',
 'Loyola (IL)',
 'Loyola Marymount',
 'Loyola (MD)',
 'Maine',
 'Manhattan',
 'Marist',
 'Marquette',
 'Marshall',
 'Maryland-Baltimore County',
 'Maryland-Eastern Shore',
 'Maryland',
 'Massachusetts-Lowell',
 'UMass',
 'McNeese State',
 'Memphis',
 'Mercer',
 'Merrimack',
 'Miami (FL)',
 'Miami (OH)',
 'Michigan State',
 'Michigan',
 'Middle Tennessee',
 'Milwaukee',
 'Minnesota',
 'Mississippi State',
 'Mississippi Valley State',
 'Mississippi',
 'Kansas City',
 'Missouri State',
 'Missouri',
 'Monmouth',
 'Montana State',
 'Montana',
 'Morehead State',
 'Morgan State',
 "Mount St. Mary's",
 'Murray State',
 'Navy',
 'Omaha',
 'Nebraska',
 'UNLV',
 'Nevada',
 'New Hampshire',
 'New Mexico State',
 'New Mexico',
 'New Mexico',
 'New Orleans',
 'Niagara',
 'Nicholls State',
 'NJIT',
 'Norfolk State',
 'North Alabama',
 'North Carolina-Asheville',
 'North Carolina A&T',
 'North Carolina Central',
 'North Carolina-Greensboro',
 'NC State',
 'UNC Wilmington',
 'UNC',
 'UNC',
 'UNC',
 'North Dakota State',
 'North Dakota',
 'North Dakota',
 'North Florida',
 'North Texas',
 'Northeastern',
 'Northern Arizona',
 'Northern Colorado',
 'Northern Illinois',
 'Northern Iowa',
 'Northern Kentucky',
 'Northwestern State',
 'Northwestern',
 'Notre Dame',
 'Oakland',
 'Ohio State',
 'Ohio',
 'Oklahoma State',
 'Oklahoma',
 'Old Dominion',
 'Oral Roberts',
 'Oregon State',
 'Oregon',
 'Pacific',
 'Penn State',
 'Pennsylvania',
 'Pepperdine',
 'Pitt',
 'Portland State',
 'Portland',
 'Prairie View',
 'Presbyterian',
 'Princeton',
 'Providence',
 'Purdue',
 'Quinnipiac',
 'Radford',
 'Rhode Island',
 'Rice',
 'Richmond',
 'Rider',
 'Robert Morris',
 'Rutgers',
 'Sacramento State',
 'Sacred Heart',
 'Saint Francis (PA)',
 "Saint Joseph's",
 'Saint Louis',
 "Saint Mary's (CA)",
 "Saint Peter's",
 'Sam Houston State',
 'Samford',
 'San Diego State',
 'San Diego',
 'San Diego',
 'San Francisco',
 'San Jose State',
 'Santa Clara',
 'Seattle',
 'Seton Hall',
 'Siena',
 'South Alabama',
 'South Carolina State',
 'South Carolina Upstate',
 'South Carolina',
 'South Carolina',
 'South Dakota State',
 'South Dakota',
 'South Dakota',
 'South Florida',
 'Southeast Missouri State',
 'Southeastern Louisiana',
 'USC',
 'SIU Edwardsville',
 'Southern Illinois',
 'SMU',
 'Southern Mississippi',
 'Southern Utah',
 'Southern',
 'St. Bonaventure',
 'St. Francis (NY)',
 "St. John's (NY)",
 'St. Thomas (MN)',
 'Stanford',
 'Stephen F. Austin',
 'Stetson',
 'Stony Brook',
 'Syracuse',
 'Tarleton State',
 'Temple',
 'Tennessee-Martin',
 'Tennessee State',
 'Tennessee Tech',
 'Tennessee',
 'Texas A&M-Corpus Christi',
 'Texas A&M',
 'Texas-Arlington',
 'TCU',
 'Texas-El Paso',
 'Texas-Rio Grande Valley',
 'Texas-San Antonio',
 'Texas Southern',
 'Texas State',
 'Texas Tech',
 'Texas',
 'Toledo',
 'Towson',
 'Troy',
 'Tulane',
 'Tulsa',
 'UCLA',
 'Utah State',
 'Utah Valley',
 'Utah',
 'Valparaiso',
 'Vanderbilt',
 'Vermont',
 'Villanova',
 'VCU',
 'VMI',
 'Virginia Tech',
 'Virginia',
 'Wagner',
 'Wake Forest',
 'Washington State',
 'Washington',
 'Weber State',
 'West Virginia',
 'Western Carolina',
 'Western Illinois',
 'Western Kentucky',
 'Western Michigan',
 'Wichita State',
 'William & Mary',
 'Winthrop',
 'Wisconsin',
 'Wofford',
 'Wright State',
 'Wyoming',
 'Xavier',
 'Yale',
 'Youngstown State']

# Defining functions

def create_current_dataframe():
    """
    Returns a dataframe of ALL data which is used by the models.
    This pulls in from both KenPom and Sportsreference
    """
    global browser,in_agg,to_change,missing,in_kp,kp_cols
    df_agg = pd.DataFrame()
    
    for team in Teams(2022):
        
        try:
            temp_df = team.dataframe
            df_agg = pd.concat([df_agg,temp_df],axis=0)
        except:
            pass
        
    #Renaming to keep naming the same
    rename_dict = dict(zip(in_agg,to_change))
    df_agg['name'] = df_agg.name.replace(rename_dict)
    return df_agg

def create_kp_df(season = 2022):
    """
    Returns a dataframe of the current KenPom standings.
    """
    kp_df2 = pd.DataFrame()
    global browser,in_agg,to_change,missing,in_kp,kp_cols
    
    temp_df = kpm.get_pomeroy_ratings(browser,season = 2022)
    temp_df.columns = kp_cols
    temp_df = temp_df.drop([temp_df.columns[i] for i in [0,2,3,6,8,10,12,14,16,18,20,19,21]],1)
    temp_df['Team'] = temp_df.Team.replace(dict(zip(in_kp,missing)))
    temp_df['Season'] = season
    kp_df2 = pd.concat([kp_df2,temp_df])
    kp_df2[kp_df2.Team != "Team"].dropna(axis = 0)
    kp_df2.Team = kp_df2.Team.str.replace(" St.",' State')
    return kp_df2

def get_game_prediction(df_agg,kp_df,home_team,away_team):
    """
    Returns:
        A predicted probability of the outcome of the game for the teams passed as parameters.
        A predicted spread for the matchup between the two teams
    
    Parameters:
        df_agg = A SportsReference aggregation dataframe.  This can be obtained by running the bb.create_current_dataframe() function
        kp_df = A KenPom summary dataframe.  This can be obtained by runningthe bb.create_kp_df() function
        home_team = A valid home team name.  A list of team names can be found by running bb.valid_team_names
        away_team = A valid away team name.  A list of team names can be found by running bb.valid_team_names
    
    
    """
    
    global wl_cols,spread_cols,wl_model,spread_model,valid_team_names,browser,missing,in_kp,kp_cols
    team_1 = home_team
    team_2 = away_team
    
    #Raise Errors
    if team_1 not in valid_team_names:
        raise ValueError("Invalid Home Team Name.  Run bb.valid_team_names to see a list of valid team names")
    if team_2 not in valid_team_names:
        raise ValueError("Invalid Away Team Name.  Run bb.valid_team_names to see a list of valid team names")
    
    #Massive data transformation/Cleaning
    if team_1 in df_agg.name.unique().tolist():
        if team_2 in df_agg.name.unique().tolist():
            temp_df_2 = df_agg.loc[df_agg.name == team_1].loc[:,df_agg.columns[df_agg.columns.str.contains('percen') | df_agg.columns.str.contains('rat') | (df_agg.columns == 'pace') | (df_agg.columns == 'name')]]
            temp_df_3 = df_agg.loc[df_agg.name == team_2].loc[:,df_agg.columns[df_agg.columns.str.contains('percen') | df_agg.columns.str.contains('rat') | (df_agg.columns == 'pace') | (df_agg.columns == 'name')]]
            old_names = temp_df_2.columns[temp_df_2.columns.str.contains('opp_')].to_list()
            new_names = [i.replace('opp','allowed') for i in temp_df_2.columns[temp_df_2.columns.str.contains('opp_')].to_list()]
            name_dict = dict(zip(old_names,new_names))
            temp_df_2_dict = temp_df_2.rename(columns = name_dict).to_dict(orient = 'list')
            temp_df_2_dict.update(temp_df_3.iloc[0,:10].rename(dict(zip(temp_df_3.iloc[0,:10].index,['opp_'+str(i) for i in temp_df_3.iloc[0,:10].index]))).to_frame().transpose().to_dict(orient = 'list'))
            temp_df_2_dict.update(temp_df_3.iloc[0,25:].rename(dict(zip(temp_df_3.iloc[0,25:].index,['opp_'+str(i) for i in temp_df_3.iloc[0,25:].index]))).to_frame().transpose().to_dict(orient = 'list'))
            temp_df_2_dict.update(temp_df_3.iloc[0,10:25].rename(dict(zip(temp_df_3.iloc[0,10:25].index,[i + "_allowed" for i in temp_df_3.iloc[0,10:25].index]))).to_frame().transpose().to_dict(orient = 'list'))
            temp_df_3_dict = temp_df_3.rename(columns = name_dict).to_dict(orient = 'list')
            temp_df_3_dict.update(temp_df_2.iloc[0,:10].rename(dict(zip(temp_df_2.iloc[0,:10].index,['opp_'+str(i) for i in temp_df_2.iloc[0,:10].index]))).to_frame().transpose().to_dict(orient = 'list'))
            temp_df_3_dict.update(temp_df_2.iloc[0,25:].rename(dict(zip(temp_df_2.iloc[0,25:].index,['opp_'+str(i) for i in temp_df_2.iloc[0,25:].index]))).to_frame().transpose().to_dict(orient = 'list'))
            temp_df_3_dict.update(temp_df_2.iloc[0,10:25].rename(dict(zip(temp_df_2.iloc[0,10:25].index,[i + "_allowed" for i in temp_df_2.iloc[0,10:25].index]))).to_frame().transpose().to_dict(orient = 'list'))
                   
    for i in temp_df_3_dict.keys():
        temp_df_2_dict[i]+=(temp_df_3_dict[i])
    
    #Create the master dictionary which will house all of the data
    master_dict = {key:[] for key in temp_df_2_dict.keys()}
    
    #Populating the dictionary
    for i in temp_df_2_dict.keys():
        master_dict[i] += temp_df_2_dict[i]
        
    #Creating a dataframe from which to make predictions
    df_pred = pd.DataFrame.from_dict(master_dict)
    df_pred = df_pred[df_pred.name == home_team]
    df_pred['home'] = 1
    
    #Ensuring features are appropriately scaled
    for i in df_pred.loc[:,df_pred.columns[df_pred.columns.str.contains('percent')]].columns:
            if df_pred[i].max() > 1:
                df_pred[i] = [j/100 for j in df_pred[i]]
    
    # Bring In kenpom data:
    kp_df2 = kp_df
    df_pred = pd.merge(df_pred,kp_df2,left_on = 'name',right_on = "Team",how = 'left').drop("Team",axis = 1)
    df_wl = df_pred[wl_cols]
    
    #Predict a win & loss probability
    loss_prob = wl_model.predict_proba(df_wl)[0][0]
    win_prob = wl_model.predict_proba(df_wl)[0][1]

    if win_prob >= loss_prob:
        print(home_team + ' has a ' + str(round(win_prob*100,2)) + '% chance of winning at home against ' +away_team)
    else:
        print(away_team + ' has a ' + str(round(loss_prob*100,2)) + '% chance of winning on the road against ' + home_team)
    
    #Predict a spread and print it
    df_spread = df_pred[spread_cols]
    predicted_spread = int(round(spread_model.predict(df_spread)[0]))
    print("Predicted Spread is {}".format(predicted_spread))


"""
To do list:
 - Add the visual option to spit out a shap value waterfall plot for the prediction.  Will need to map colors per school
    - This will also require pickling the shap.explainer object for the w_l model, which will require re-training
 - Clean up output of the dataframe function
- Try training the models without the simple rating system - it may be a redundant feature.
- Try training the models with the KenPom four factors data.
"""