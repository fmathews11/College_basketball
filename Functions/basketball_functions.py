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
f = open('neutral_court_model.pickle','rb')
neutral_court_model = pickle.load(f)

#Create Kenpom browser 
browser = login(kp_login, kp_pw)
sum_df = kp.get_efficiency(browser,2022)
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

neutral_court_cols = ['offensive_rating',
 'allowed_effective_field_goal_percentage',
 'allowed_total_rebound_percentage',
 'allowed_turnover_percentage',
 'simple_rating_system',
 'total_rebound_percentage',
 'win_percentage',
 'opp_assist_percentage',
 'opp_offensive_rating',
 'opp_simple_rating_system',
 'opp_win_percentage',
 'opp_effective_field_goal_percentage_allowed',
 'opp_free_throw_attempt_rate_allowed',
 'opp_free_throw_percentage_allowed',
 'opp_turnover_percentage_allowed',
 'Luck']

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

list_of_stats = ['team',
                 'home',
                 'opponent',
                 'date',
                 'season',
                 'pace',
                  'result',
                  'wins',
                  'win_percentage',
                  'two_point_field_goals',
                  'two_point_field_goal_percentage',
                   'two_point_field_goal_attempts',
                   'turnovers',
                   'turnover_percentage',
                    'true_shooting_percentage',
                    'total_rebounds',
                    'total_rebound_percentage',
                     'three_point_field_goals',
                     'three_point_field_goal_percentage',
                     'three_point_field_goal_attempts',
                     'three_point_attempt_rate',
                     'steals','steal_percentage',
                     'ranking',
                     'points',
                     'personal_fouls',
                     'offensive_rebounds',
                     'offensive_rebound_percentage',
                     'offensive_rating',
                     'minutes_played',
                     'losses',
                     'free_throws',
                     'free_throw_percentage',
                     'free_throw_attempts',
                     'free_throw_attempt_rate',
                     'field_goals',
                     'field_goal_percentage',
                     'field_goal_attempts',
                     'effective_field_goal_percentage',
                     'defensive_rebounds',
                     'defensive_rating',
                     'blocks',
                     'block_percentage',
                     'assists',
                     'assist_percentage']
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

def get_game_prediction(df_agg,kp_df,home_team,away_team,spread = True,verbose = False):
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
    
    if verbose:
        if win_prob >= loss_prob:
            print(home_team + ' has a ' + str(round(win_prob*100,2)) + '% chance of winning at home against ' +away_team)
        else:
            print(away_team + ' has a ' + str(round(loss_prob*100,2)) + '% chance of winning on the road against ' + home_team)
    
    if spread:
    #Predict a spread and print it
        df_spread = df_pred[spread_cols]
        predicted_spread = int(round(spread_model.predict(df_spread)[0]))
        if verbose:
            print("Predicted Spread is {}".format(predicted_spread))
        return round(win_prob,3),predicted_spread
    else:
        return round(win_prob,3)

    
def get_game_prediction_neutral(df_agg,kp_df,team_1,team_2):

# Win probability is expressed in the form of team_1 beating team_2 on a neutral court

    class CustomError(Exception):
        pass

    if team_1 in df_agg.name.unique().tolist() and team_2 in df_agg.name.unique().tolist():
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
        master_dict = {key:[] for key in temp_df_2_dict.keys()}

        for i in temp_df_2_dict.keys():
            master_dict[i] += temp_df_2_dict[i]
        df_pred = pd.DataFrame.from_dict(master_dict)
        df_pred = df_pred[df_pred.name == team_1]
        for i in df_pred.loc[:,df_pred.columns[df_pred.columns.str.contains('percent')]].columns:
             if df_pred[i].max() > 1:
                df_pred[i] = [j/100 for j in df_pred[i]]
        df_pred = pd.merge(df_pred,kp_df,left_on = 'name',right_on = "Team",how = 'left').drop("Team",axis = 1)
        return round(neutral_court_model.predict_proba(df_pred[neutral_court_cols])[0][1],3)
    else:
        raise CustomError("Check Team Names")    

    
def transform_box_df(df):

#Establishing a list of stats which I want
    list_of_stats = ['team',
                     'home',
                     'opponent',
                     'date',
                     'season',
                     'pace',
                     'result',
                     'wins',
                     'win_percentage',
                     'two_point_field_goals',
                     'two_point_field_goal_percentage',
                     'two_point_field_goal_attempts',
                     'turnovers',
                     'turnover_percentage',
                     'true_shooting_percentage',
                     'total_rebounds',
                     'total_rebound_percentage',
                     'three_point_field_goals',
                     'three_point_field_goal_percentage',
                     'three_point_field_goal_attempts',
                     'three_point_attempt_rate',
                     'steals','steal_percentage',
                     'ranking',
                     'points',
                     'personal_fouls',
                     'offensive_rebounds',
                     'offensive_rebound_percentage',
                     'offensive_rating',
                     'minutes_played',
                     'losses',
                     'free_throws',
                     'free_throw_percentage',
                     'free_throw_attempts',
                     'free_throw_attempt_rate',
                     'field_goals',
                     'field_goal_percentage',
                     'field_goal_attempts',
                     'effective_field_goal_percentage',
                     'defensive_rebounds',
                     'defensive_rating',
                     'blocks',
                     'block_percentage',
                     'assists',
                     'assist_percentage']

# Adding opponent_ for all the opponent stats
    for stat in list_of_stats[7:]:
        list_of_stats.append('opponent_'+stat)
        
#Creating an empty dictionary to populate.  This approach is much faster than using Pandas    
    stats_dict = {key:[] for key in list_of_stats}
    
#Begin the loop
    for Team in df.losing_name.unique().tolist():

        #Home/Winner
        temp_df = df[(df.winning_name == Team) & (df.winner == "Home")]
        stats_dict['team'] += temp_df.winning_name.tolist()
        stats_dict['opponent']  += temp_df.losing_name.tolist()
        stats_dict['date']  += temp_df.date.tolist()
        stats_dict['pace']  += temp_df.pace.tolist()
        stats_dict['wins']  += temp_df.home_wins.tolist()
        stats_dict['win_percentage']  += temp_df.home_win_percentage.tolist()
        stats_dict['two_point_field_goals']  += temp_df.home_two_point_field_goals.tolist()
        stats_dict['two_point_field_goal_percentage']  += temp_df.home_two_point_field_goal_percentage.tolist() 
        stats_dict['two_point_field_goal_attempts']  += temp_df.home_two_point_field_goal_attempts.tolist()
        stats_dict['turnovers']  += temp_df.home_turnovers.tolist()
        stats_dict['turnover_percentage']  += temp_df.home_turnover_percentage.tolist()
        stats_dict['true_shooting_percentage']  += temp_df.home_true_shooting_percentage.tolist()
        stats_dict['total_rebounds']  += temp_df.home_total_rebounds.tolist()
        stats_dict['total_rebound_percentage']  += temp_df.home_total_rebound_percentage.tolist()
        stats_dict['three_point_field_goals']  += temp_df.home_three_point_field_goals.tolist()
        stats_dict['three_point_field_goal_percentage']  += temp_df.home_three_point_field_goal_percentage.tolist()
        stats_dict['three_point_field_goal_attempts']  += temp_df.home_three_point_field_goal_attempts.tolist()
        stats_dict['three_point_attempt_rate']  += temp_df.home_three_point_attempt_rate.tolist()
        stats_dict['steals']  += temp_df.home_steals.tolist()
        stats_dict['steal_percentage']  += temp_df.home_steal_percentage.tolist()
        stats_dict['ranking']  += temp_df.home_ranking.tolist()
        stats_dict['points']  += temp_df.home_points.tolist()
        stats_dict['personal_fouls']  += temp_df.home_personal_fouls.tolist()
        stats_dict['offensive_rebounds']  += temp_df.home_offensive_rebounds.tolist()
        stats_dict['offensive_rebound_percentage']  += temp_df.home_offensive_rebound_percentage.tolist()
        stats_dict['offensive_rating']  += temp_df.home_offensive_rating.tolist()
        stats_dict['minutes_played']  += temp_df.home_minutes_played.tolist()
        stats_dict['losses']  += temp_df.home_losses.tolist()
        stats_dict['free_throws']  += temp_df.home_free_throws.tolist()
        stats_dict['free_throw_percentage']  += temp_df.home_free_throw_percentage.tolist()
        stats_dict['free_throw_attempts']  += temp_df.home_free_throw_attempts.tolist()
        stats_dict['free_throw_attempt_rate']  += temp_df.home_free_throw_attempt_rate.tolist()
        stats_dict['field_goals']  += temp_df.home_field_goals.tolist()
        stats_dict['field_goal_percentage']  += temp_df.home_field_goal_percentage.tolist()
        stats_dict['field_goal_attempts']  += temp_df.home_field_goal_attempts.tolist()
        stats_dict['effective_field_goal_percentage']  += temp_df.home_effective_field_goal_percentage.tolist()
        stats_dict['defensive_rebounds']  += temp_df.home_defensive_rebounds.tolist()
        stats_dict['defensive_rating']  += temp_df.home_defensive_rating.tolist()
        stats_dict['blocks']  += temp_df.home_blocks.tolist()
        stats_dict['block_percentage']  += temp_df.home_block_percentage.tolist()
        stats_dict['assists']  += temp_df.home_assists.tolist()
        stats_dict['assist_percentage']  += temp_df.home_assist_percentage.tolist()
        stats_dict['result'] += [1] * len(temp_df)
        stats_dict['opponent_wins'] += temp_df.away_wins.tolist()
        stats_dict['opponent_win_percentage']  += temp_df.away_win_percentage.tolist()
        stats_dict['opponent_two_point_field_goals']  += temp_df.away_two_point_field_goals.tolist()
        stats_dict['opponent_two_point_field_goal_percentage']  += temp_df.away_two_point_field_goal_percentage.tolist() 
        stats_dict['opponent_two_point_field_goal_attempts']  += temp_df.away_two_point_field_goal_attempts.tolist()
        stats_dict['opponent_turnovers']  += temp_df.away_turnovers.tolist()
        stats_dict['opponent_turnover_percentage']  += temp_df.away_turnover_percentage.tolist()
        stats_dict['opponent_true_shooting_percentage']  += temp_df.away_true_shooting_percentage.tolist()
        stats_dict['opponent_total_rebounds']  += temp_df.away_total_rebounds.tolist()
        stats_dict['opponent_total_rebound_percentage']  += temp_df.away_total_rebound_percentage.tolist()
        stats_dict['opponent_three_point_field_goals']  += temp_df.away_three_point_field_goals.tolist()
        stats_dict['opponent_three_point_field_goal_percentage']  += temp_df.away_three_point_field_goal_percentage.tolist()
        stats_dict['opponent_three_point_field_goal_attempts']  += temp_df.away_three_point_field_goal_attempts.tolist()
        stats_dict['opponent_three_point_attempt_rate']  += temp_df.away_three_point_attempt_rate.tolist()
        stats_dict['opponent_steals']  += temp_df.away_steals.tolist()
        stats_dict['opponent_steal_percentage']  += temp_df.away_steal_percentage.tolist()
        stats_dict['opponent_ranking']  += temp_df.away_ranking.tolist()
        stats_dict['opponent_points']  += temp_df.away_points.tolist()
        stats_dict['opponent_personal_fouls']  += temp_df.away_personal_fouls.tolist()
        stats_dict['opponent_offensive_rebounds']  += temp_df.away_offensive_rebounds.tolist()
        stats_dict['opponent_offensive_rebound_percentage']  += temp_df.away_offensive_rebound_percentage.tolist()
        stats_dict['opponent_offensive_rating']  += temp_df.away_offensive_rating.tolist()
        stats_dict['opponent_minutes_played']  += temp_df.away_minutes_played.tolist()
        stats_dict['opponent_losses']  += temp_df.away_losses.tolist()
        stats_dict['opponent_free_throws']  += temp_df.away_free_throws.tolist()
        stats_dict['opponent_free_throw_percentage']  += temp_df.away_free_throw_percentage.tolist()
        stats_dict['opponent_free_throw_attempts']  += temp_df.away_free_throw_attempts.tolist()
        stats_dict['opponent_free_throw_attempt_rate']  += temp_df.away_free_throw_attempt_rate.tolist()
        stats_dict['opponent_field_goals']  += temp_df.away_field_goals.tolist()
        stats_dict['opponent_field_goal_percentage']  += temp_df.away_field_goal_percentage.tolist()
        stats_dict['opponent_field_goal_attempts']  += temp_df.away_field_goal_attempts.tolist()
        stats_dict['opponent_effective_field_goal_percentage']  += temp_df.away_effective_field_goal_percentage.tolist()
        stats_dict['opponent_defensive_rebounds']  += temp_df.away_defensive_rebounds.tolist()
        stats_dict['opponent_defensive_rating']  += temp_df.away_defensive_rating.tolist()
        stats_dict['opponent_blocks']  += temp_df.away_blocks.tolist()
        stats_dict['opponent_block_percentage']  += temp_df.away_block_percentage.tolist()
        stats_dict['opponent_assists']  += temp_df.away_assists.tolist()
        stats_dict['opponent_assist_percentage']  += temp_df.away_assist_percentage.tolist()
        stats_dict['home'] += [1] * len(temp_df)
        stats_dict['season']  += temp_df.season.tolist()

        #Away/Winner  Games won on the road
        temp_df = df[(df.winning_name == Team) & (df.winner == "Away")]
        stats_dict['team']  += temp_df.winning_name.tolist()
        stats_dict['opponent']  += temp_df.losing_name.tolist()
        stats_dict['date']  += temp_df.date.tolist()
        stats_dict['pace']  += temp_df.pace.tolist()
        stats_dict['wins']  += temp_df.away_wins.tolist()
        stats_dict['win_percentage']  += temp_df.away_win_percentage.tolist()
        stats_dict['two_point_field_goals']  += temp_df.away_two_point_field_goals.tolist()
        stats_dict['two_point_field_goal_percentage']  += temp_df.away_two_point_field_goal_percentage.tolist() 
        stats_dict['two_point_field_goal_attempts']  += temp_df.away_two_point_field_goal_attempts.tolist()
        stats_dict['turnovers']  += temp_df.away_turnovers.tolist()
        stats_dict['turnover_percentage']  += temp_df.away_turnover_percentage.tolist()
        stats_dict['true_shooting_percentage']  += temp_df.away_true_shooting_percentage.tolist()
        stats_dict['total_rebounds']  += temp_df.away_total_rebounds.tolist()
        stats_dict['total_rebound_percentage']  += temp_df.away_total_rebound_percentage.tolist()
        stats_dict['three_point_field_goals']  += temp_df.away_three_point_field_goals.tolist()
        stats_dict['three_point_field_goal_percentage']  += temp_df.away_three_point_field_goal_percentage.tolist()
        stats_dict['three_point_field_goal_attempts']  += temp_df.away_three_point_field_goal_attempts.tolist()
        stats_dict['three_point_attempt_rate']  += temp_df.away_three_point_attempt_rate.tolist()
        stats_dict['steals']  += temp_df.away_steals.tolist()
        stats_dict['steal_percentage']  += temp_df.away_steal_percentage.tolist()
        stats_dict['ranking']  += temp_df.away_ranking.tolist()
        stats_dict['points']  += temp_df.away_points.tolist()
        stats_dict['personal_fouls']  += temp_df.away_personal_fouls.tolist()
        stats_dict['offensive_rebounds']  += temp_df.away_offensive_rebounds.tolist()
        stats_dict['offensive_rebound_percentage']  += temp_df.away_offensive_rebound_percentage.tolist()
        stats_dict['offensive_rating']  += temp_df.away_offensive_rating.tolist()
        stats_dict['minutes_played']  += temp_df.away_minutes_played.tolist()
        stats_dict['losses']  += temp_df.away_losses.tolist()
        stats_dict['free_throws']  += temp_df.away_free_throws.tolist()
        stats_dict['free_throw_percentage']  += temp_df.away_free_throw_percentage.tolist()
        stats_dict['free_throw_attempts']  += temp_df.away_free_throw_attempts.tolist()
        stats_dict['free_throw_attempt_rate']  += temp_df.away_free_throw_attempt_rate.tolist()
        stats_dict['field_goals']  += temp_df.away_field_goals.tolist()
        stats_dict['field_goal_percentage']  += temp_df.away_field_goal_percentage.tolist()
        stats_dict['field_goal_attempts']  += temp_df.away_field_goal_attempts.tolist()
        stats_dict['effective_field_goal_percentage']  += temp_df.away_effective_field_goal_percentage.tolist()
        stats_dict['defensive_rebounds']  += temp_df.away_defensive_rebounds.tolist()
        stats_dict['defensive_rating']  += temp_df.away_defensive_rating.tolist()
        stats_dict['blocks']  += temp_df.away_blocks.tolist()
        stats_dict['block_percentage']  += temp_df.away_block_percentage.tolist()
        stats_dict['assists']  += temp_df.away_assists.tolist()
        stats_dict['assist_percentage']  += temp_df.away_assist_percentage.tolist()
        stats_dict['result'] += [1] * len(temp_df)
        stats_dict['opponent_wins']  += temp_df.home_wins.tolist()
        stats_dict['opponent_win_percentage']  += temp_df.home_win_percentage.tolist()
        stats_dict['opponent_two_point_field_goals']  += temp_df.home_two_point_field_goals.tolist()
        stats_dict['opponent_two_point_field_goal_percentage']  += temp_df.home_two_point_field_goal_percentage.tolist() 
        stats_dict['opponent_two_point_field_goal_attempts']  += temp_df.home_two_point_field_goal_attempts.tolist()
        stats_dict['opponent_turnovers']  += temp_df.home_turnovers.tolist()
        stats_dict['opponent_turnover_percentage']  += temp_df.home_turnover_percentage.tolist()
        stats_dict['opponent_true_shooting_percentage']  += temp_df.home_true_shooting_percentage.tolist()
        stats_dict['opponent_total_rebounds']  += temp_df.home_total_rebounds.tolist()
        stats_dict['opponent_total_rebound_percentage']  += temp_df.home_total_rebound_percentage.tolist()
        stats_dict['opponent_three_point_field_goals']  += temp_df.home_three_point_field_goals.tolist()
        stats_dict['opponent_three_point_field_goal_percentage']  += temp_df.home_three_point_field_goal_percentage.tolist()
        stats_dict['opponent_three_point_field_goal_attempts']  += temp_df.home_three_point_field_goal_attempts.tolist()
        stats_dict['opponent_three_point_attempt_rate']  += temp_df.home_three_point_attempt_rate.tolist()
        stats_dict['opponent_steals']  += temp_df.home_steals.tolist()
        stats_dict['opponent_steal_percentage']  += temp_df.home_steal_percentage.tolist()
        stats_dict['opponent_ranking']  += temp_df.home_ranking.tolist()
        stats_dict['opponent_points']  += temp_df.home_points.tolist()
        stats_dict['opponent_personal_fouls']  += temp_df.home_personal_fouls.tolist()
        stats_dict['opponent_offensive_rebounds']  += temp_df.home_offensive_rebounds.tolist()
        stats_dict['opponent_offensive_rebound_percentage']  += temp_df.home_offensive_rebound_percentage.tolist()
        stats_dict['opponent_offensive_rating']  += temp_df.home_offensive_rating.tolist()
        stats_dict['opponent_minutes_played']  += temp_df.home_minutes_played.tolist()
        stats_dict['opponent_losses']  += temp_df.home_losses.tolist()
        stats_dict['opponent_free_throws']  += temp_df.home_free_throws.tolist()
        stats_dict['opponent_free_throw_percentage']  += temp_df.home_free_throw_percentage.tolist()
        stats_dict['opponent_free_throw_attempts']  += temp_df.home_free_throw_attempts.tolist()
        stats_dict['opponent_free_throw_attempt_rate']  += temp_df.home_free_throw_attempt_rate.tolist()
        stats_dict['opponent_field_goals']  += temp_df.home_field_goals.tolist()
        stats_dict['opponent_field_goal_percentage']  += temp_df.home_field_goal_percentage.tolist()
        stats_dict['opponent_field_goal_attempts']  += temp_df.home_field_goal_attempts.tolist()
        stats_dict['opponent_effective_field_goal_percentage']  += temp_df.home_effective_field_goal_percentage.tolist()
        stats_dict['opponent_defensive_rebounds']  += temp_df.home_defensive_rebounds.tolist()
        stats_dict['opponent_defensive_rating']  += temp_df.home_defensive_rating.tolist()
        stats_dict['opponent_blocks']  += temp_df.home_blocks.tolist()
        stats_dict['opponent_block_percentage']  += temp_df.home_block_percentage.tolist()
        stats_dict['opponent_assists']  += temp_df.home_assists.tolist()
        stats_dict['opponent_assist_percentage']  += temp_df.home_assist_percentage.tolist()
        stats_dict['home'] += [0] * len(temp_df)
        stats_dict['season']  += temp_df.season.tolist()

        #Away/loser Games lost at home
        temp_df = df[(df.losing_name == Team) & (df.winner == "Away")]
        stats_dict['team']  += temp_df.losing_name.tolist()
        stats_dict['opponent']  += temp_df.winning_name.tolist()
        stats_dict['date']  += temp_df.date.tolist()
        stats_dict['pace']  += temp_df.pace.tolist()
        stats_dict['wins']  += temp_df.home_wins.tolist()
        stats_dict['win_percentage']  += temp_df.home_win_percentage.tolist()
        stats_dict['two_point_field_goals']  += temp_df.home_two_point_field_goals.tolist()
        stats_dict['two_point_field_goal_percentage']  += temp_df.home_two_point_field_goal_percentage.tolist() 
        stats_dict['two_point_field_goal_attempts']  += temp_df.home_two_point_field_goal_attempts.tolist()
        stats_dict['turnovers']  += temp_df.home_turnovers.tolist()
        stats_dict['turnover_percentage']  += temp_df.home_turnover_percentage.tolist()
        stats_dict['true_shooting_percentage']  += temp_df.home_true_shooting_percentage.tolist()
        stats_dict['total_rebounds']  += temp_df.home_total_rebounds.tolist()
        stats_dict['total_rebound_percentage']  += temp_df.home_total_rebound_percentage.tolist()
        stats_dict['three_point_field_goals']  += temp_df.home_three_point_field_goals.tolist()
        stats_dict['three_point_field_goal_percentage']  += temp_df.home_three_point_field_goal_percentage.tolist()
        stats_dict['three_point_field_goal_attempts']  += temp_df.home_three_point_field_goal_attempts.tolist()
        stats_dict['three_point_attempt_rate']  += temp_df.home_three_point_attempt_rate.tolist()
        stats_dict['steals']  += temp_df.home_steals.tolist()
        stats_dict['steal_percentage']  += temp_df.home_steal_percentage.tolist()
        stats_dict['ranking']  += temp_df.home_ranking.tolist()
        stats_dict['points']  += temp_df.home_points.tolist()
        stats_dict['personal_fouls']  += temp_df.home_personal_fouls.tolist()
        stats_dict['offensive_rebounds']  += temp_df.home_offensive_rebounds.tolist()
        stats_dict['offensive_rebound_percentage']  += temp_df.home_offensive_rebound_percentage.tolist()
        stats_dict['offensive_rating']  += temp_df.home_offensive_rating.tolist()
        stats_dict['minutes_played']  += temp_df.home_minutes_played.tolist()
        stats_dict['losses']  += temp_df.home_losses.tolist()
        stats_dict['free_throws']  += temp_df.home_free_throws.tolist()
        stats_dict['free_throw_percentage']  += temp_df.home_free_throw_percentage.tolist()
        stats_dict['free_throw_attempts']  += temp_df.home_free_throw_attempts.tolist()
        stats_dict['free_throw_attempt_rate']  += temp_df.home_free_throw_attempt_rate.tolist()
        stats_dict['field_goals']  += temp_df.home_field_goals.tolist()
        stats_dict['field_goal_percentage']  += temp_df.home_field_goal_percentage.tolist()
        stats_dict['field_goal_attempts']  += temp_df.home_field_goal_attempts.tolist()
        stats_dict['effective_field_goal_percentage']  += temp_df.home_effective_field_goal_percentage.tolist()
        stats_dict['defensive_rebounds']  += temp_df.home_defensive_rebounds.tolist()
        stats_dict['defensive_rating']  += temp_df.home_defensive_rating.tolist()
        stats_dict['blocks']  += temp_df.home_blocks.tolist()
        stats_dict['block_percentage']  += temp_df.home_block_percentage.tolist()
        stats_dict['assists']  += temp_df.home_assists.tolist()
        stats_dict['assist_percentage']  += temp_df.home_assist_percentage.tolist()
        stats_dict['result'] += [0] * len(temp_df)
        stats_dict['opponent_wins']  += temp_df.away_wins.tolist()
        stats_dict['opponent_win_percentage']  += temp_df.away_win_percentage.tolist()
        stats_dict['opponent_two_point_field_goals']  += temp_df.away_two_point_field_goals.tolist()
        stats_dict['opponent_two_point_field_goal_percentage']  += temp_df.away_two_point_field_goal_percentage.tolist() 
        stats_dict['opponent_two_point_field_goal_attempts']  += temp_df.away_two_point_field_goal_attempts.tolist()
        stats_dict['opponent_turnovers']  += temp_df.away_turnovers.tolist()
        stats_dict['opponent_turnover_percentage']  += temp_df.away_turnover_percentage.tolist()
        stats_dict['opponent_true_shooting_percentage']  += temp_df.away_true_shooting_percentage.tolist()
        stats_dict['opponent_total_rebounds']  += temp_df.away_total_rebounds.tolist()
        stats_dict['opponent_total_rebound_percentage']  += temp_df.away_total_rebound_percentage.tolist()
        stats_dict['opponent_three_point_field_goals']  += temp_df.away_three_point_field_goals.tolist()
        stats_dict['opponent_three_point_field_goal_percentage']  += temp_df.away_three_point_field_goal_percentage.tolist()
        stats_dict['opponent_three_point_field_goal_attempts']  += temp_df.away_three_point_field_goal_attempts.tolist()
        stats_dict['opponent_three_point_attempt_rate']  += temp_df.away_three_point_attempt_rate.tolist()
        stats_dict['opponent_steals']  += temp_df.away_steals.tolist()
        stats_dict['opponent_steal_percentage']  += temp_df.away_steal_percentage.tolist()
        stats_dict['opponent_ranking']  += temp_df.away_ranking.tolist()
        stats_dict['opponent_points']  += temp_df.away_points.tolist()
        stats_dict['opponent_personal_fouls']  += temp_df.away_personal_fouls.tolist()
        stats_dict['opponent_offensive_rebounds']  += temp_df.away_offensive_rebounds.tolist()
        stats_dict['opponent_offensive_rebound_percentage']  += temp_df.away_offensive_rebound_percentage.tolist()
        stats_dict['opponent_offensive_rating']  += temp_df.away_offensive_rating.tolist()
        stats_dict['opponent_minutes_played']  += temp_df.away_minutes_played.tolist()
        stats_dict['opponent_losses']  += temp_df.away_losses.tolist()
        stats_dict['opponent_free_throws']  += temp_df.away_free_throws.tolist()
        stats_dict['opponent_free_throw_percentage']  += temp_df.away_free_throw_percentage.tolist()
        stats_dict['opponent_free_throw_attempts']  += temp_df.away_free_throw_attempts.tolist()
        stats_dict['opponent_free_throw_attempt_rate']  += temp_df.away_free_throw_attempt_rate.tolist()
        stats_dict['opponent_field_goals']  += temp_df.away_field_goals.tolist()
        stats_dict['opponent_field_goal_percentage']  += temp_df.away_field_goal_percentage.tolist()
        stats_dict['opponent_field_goal_attempts']  += temp_df.away_field_goal_attempts.tolist()
        stats_dict['opponent_effective_field_goal_percentage']  += temp_df.away_effective_field_goal_percentage.tolist()
        stats_dict['opponent_defensive_rebounds']  += temp_df.away_defensive_rebounds.tolist()
        stats_dict['opponent_defensive_rating']  += temp_df.away_defensive_rating.tolist()
        stats_dict['opponent_blocks']  += temp_df.away_blocks.tolist()
        stats_dict['opponent_block_percentage']  += temp_df.away_block_percentage.tolist()
        stats_dict['opponent_assists']  += temp_df.away_assists.tolist()
        stats_dict['opponent_assist_percentage']  += temp_df.away_assist_percentage.tolist()
        stats_dict['home'] += [1] * len(temp_df)
        stats_dict['season']  += temp_df.season.tolist()

        #Home/loser Games lost on the road
        temp_df = df[(df.losing_name == Team) & (df.winner == "Home")]
        stats_dict['team']  += temp_df.losing_name.tolist()
        stats_dict['opponent']  += temp_df.winning_name.tolist()
        stats_dict['date']  += temp_df.date.tolist()
        stats_dict['pace']  += temp_df.pace.tolist()
        stats_dict['wins']  += temp_df.away_wins.tolist()
        stats_dict['win_percentage']  += temp_df.away_win_percentage.tolist()
        stats_dict['two_point_field_goals']  += temp_df.away_two_point_field_goals.tolist()
        stats_dict['two_point_field_goal_percentage']  += temp_df.away_two_point_field_goal_percentage.tolist() 
        stats_dict['two_point_field_goal_attempts']  += temp_df.away_two_point_field_goal_attempts.tolist()
        stats_dict['turnovers']  += temp_df.away_turnovers.tolist()
        stats_dict['turnover_percentage']  += temp_df.away_turnover_percentage.tolist()
        stats_dict['true_shooting_percentage']  += temp_df.away_true_shooting_percentage.tolist()
        stats_dict['total_rebounds']  += temp_df.away_total_rebounds.tolist()
        stats_dict['total_rebound_percentage']  += temp_df.away_total_rebound_percentage.tolist()
        stats_dict['three_point_field_goals']  += temp_df.away_three_point_field_goals.tolist()
        stats_dict['three_point_field_goal_percentage']  += temp_df.away_three_point_field_goal_percentage.tolist()
        stats_dict['three_point_field_goal_attempts']  += temp_df.away_three_point_field_goal_attempts.tolist()
        stats_dict['three_point_attempt_rate']  += temp_df.away_three_point_attempt_rate.tolist()
        stats_dict['steals']  += temp_df.away_steals.tolist()
        stats_dict['steal_percentage']  += temp_df.away_steal_percentage.tolist()
        stats_dict['ranking']  += temp_df.away_ranking.tolist()
        stats_dict['points']  += temp_df.away_points.tolist()
        stats_dict['personal_fouls']  += temp_df.away_personal_fouls.tolist()
        stats_dict['offensive_rebounds']  += temp_df.away_offensive_rebounds.tolist()
        stats_dict['offensive_rebound_percentage']  += temp_df.away_offensive_rebound_percentage.tolist()
        stats_dict['offensive_rating']  += temp_df.away_offensive_rating.tolist()
        stats_dict['minutes_played']  += temp_df.away_minutes_played.tolist()
        stats_dict['losses']  += temp_df.away_losses.tolist()
        stats_dict['free_throws']  += temp_df.away_free_throws.tolist()
        stats_dict['free_throw_percentage']  += temp_df.away_free_throw_percentage.tolist()
        stats_dict['free_throw_attempts']  += temp_df.away_free_throw_attempts.tolist()
        stats_dict['free_throw_attempt_rate']  += temp_df.away_free_throw_attempt_rate.tolist()
        stats_dict['field_goals']  += temp_df.away_field_goals.tolist()
        stats_dict['field_goal_percentage']  += temp_df.away_field_goal_percentage.tolist()
        stats_dict['field_goal_attempts']  += temp_df.away_field_goal_attempts.tolist()
        stats_dict['effective_field_goal_percentage']  += temp_df.away_effective_field_goal_percentage.tolist()
        stats_dict['defensive_rebounds']  += temp_df.away_defensive_rebounds.tolist()
        stats_dict['defensive_rating']  += temp_df.away_defensive_rating.tolist()
        stats_dict['blocks']  += temp_df.away_blocks.tolist()
        stats_dict['block_percentage']  += temp_df.away_block_percentage.tolist()
        stats_dict['assists']  += temp_df.away_assists.tolist()
        stats_dict['assist_percentage']  += temp_df.away_assist_percentage.tolist()
        stats_dict['result'] += [0] * len(temp_df)
        stats_dict['opponent_wins']  += temp_df.home_wins.tolist()
        stats_dict['opponent_win_percentage']  += temp_df.home_win_percentage.tolist()
        stats_dict['opponent_two_point_field_goals']  += temp_df.home_two_point_field_goals.tolist()
        stats_dict['opponent_two_point_field_goal_percentage']  += temp_df.home_two_point_field_goal_percentage.tolist() 
        stats_dict['opponent_two_point_field_goal_attempts']  += temp_df.home_two_point_field_goal_attempts.tolist()
        stats_dict['opponent_turnovers']  += temp_df.home_turnovers.tolist()
        stats_dict['opponent_turnover_percentage']  += temp_df.home_turnover_percentage.tolist()
        stats_dict['opponent_true_shooting_percentage']  += temp_df.home_true_shooting_percentage.tolist()
        stats_dict['opponent_total_rebounds']  += temp_df.home_total_rebounds.tolist()
        stats_dict['opponent_total_rebound_percentage']  += temp_df.home_total_rebound_percentage.tolist()
        stats_dict['opponent_three_point_field_goals']  += temp_df.home_three_point_field_goals.tolist()
        stats_dict['opponent_three_point_field_goal_percentage']  += temp_df.home_three_point_field_goal_percentage.tolist()
        stats_dict['opponent_three_point_field_goal_attempts']  += temp_df.home_three_point_field_goal_attempts.tolist()
        stats_dict['opponent_three_point_attempt_rate']  += temp_df.home_three_point_attempt_rate.tolist()
        stats_dict['opponent_steals']  += temp_df.home_steals.tolist()
        stats_dict['opponent_steal_percentage']  += temp_df.home_steal_percentage.tolist()
        stats_dict['opponent_ranking']  += temp_df.home_ranking.tolist()
        stats_dict['opponent_points']  += temp_df.home_points.tolist()
        stats_dict['opponent_personal_fouls']  += temp_df.home_personal_fouls.tolist()
        stats_dict['opponent_offensive_rebounds']  += temp_df.home_offensive_rebounds.tolist()
        stats_dict['opponent_offensive_rebound_percentage']  += temp_df.home_offensive_rebound_percentage.tolist()
        stats_dict['opponent_offensive_rating']  += temp_df.home_offensive_rating.tolist()
        stats_dict['opponent_minutes_played']  += temp_df.home_minutes_played.tolist()
        stats_dict['opponent_losses']  += temp_df.home_losses.tolist()
        stats_dict['opponent_free_throws']  += temp_df.home_free_throws.tolist()
        stats_dict['opponent_free_throw_percentage']  += temp_df.home_free_throw_percentage.tolist()
        stats_dict['opponent_free_throw_attempts']  += temp_df.home_free_throw_attempts.tolist()
        stats_dict['opponent_free_throw_attempt_rate']  += temp_df.home_free_throw_attempt_rate.tolist()
        stats_dict['opponent_field_goals']  += temp_df.home_field_goals.tolist()
        stats_dict['opponent_field_goal_percentage']  += temp_df.home_field_goal_percentage.tolist()
        stats_dict['opponent_field_goal_attempts']  += temp_df.home_field_goal_attempts.tolist()
        stats_dict['opponent_effective_field_goal_percentage']  += temp_df.home_effective_field_goal_percentage.tolist()
        stats_dict['opponent_defensive_rebounds']  += temp_df.home_defensive_rebounds.tolist()
        stats_dict['opponent_defensive_rating']  += temp_df.home_defensive_rating.tolist()
        stats_dict['opponent_blocks']  += temp_df.home_blocks.tolist()
        stats_dict['opponent_block_percentage']  += temp_df.home_block_percentage.tolist()
        stats_dict['opponent_assists']  += temp_df.home_assists.tolist()
        stats_dict['opponent_assist_percentage']  += temp_df.home_assist_percentage.tolist()
        stats_dict['home'] += [0] * len(temp_df)
        stats_dict['season']  += temp_df.season.tolist()
  #Conver the dicionary to a pd.DataFrame() and do some brief cleanup      
    final_df = pd.DataFrame.from_dict(stats_dict)
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df.drop_duplicates(inplace = True)
    final_df.sort_values('date',inplace = True)
    
    return final_df

def get_expected_spread(sum_df,team_1,team_2,verbose = False,neutral = True):

    for i in sum_df.columns:
        try:
            sum_df[i] = pd.to_numeric(sum_df[i])
        except:
            pass
    sum_df.columns = ['_'.join(i.split()) for i in sum_df.columns.tolist()]
    sum_df.columns = [i.replace(".","") for i in sum_df.columns]
    sum_df.columns = [i.replace("-","_") for i in sum_df.columns]
    off_eff_avg = round(sum_df.Off_Efficiency_Adj.mean(),2)
    def_eff_avg = round(sum_df.Def_Efficiency_Adj.mean(),2)

    team_1_off_eff = sum_df.loc[sum_df.Team == team_1].Off_Efficiency_Adj.item()
    team_1_def_eff = sum_df.loc[sum_df.Team == team_1].Def_Efficiency_Adj.item()
    team_2_off_eff = sum_df.loc[sum_df.Team == team_2].Off_Efficiency_Adj.item()
    team_2_def_eff = sum_df.loc[sum_df.Team == team_2].Def_Efficiency_Adj.item()
    team_1_tempo = sum_df.loc[sum_df.Team == team_1].Tempo_Adj.item()
    team_2_tempo = sum_df.loc[sum_df.Team == team_2].Tempo_Adj.item()

    team_1_off_pct = (team_1_off_eff/off_eff_avg)-1
    team_1_off_pct_adjustment_factor = 1+round(team_1_off_pct + ((team_2_def_eff/def_eff_avg) - 1),2)
    team_1_off_adj = round(off_eff_avg * team_1_off_pct_adjustment_factor,2)
    

    team_2_off_pct = (team_2_off_eff/off_eff_avg)-1
    team_2_off_pct_adjustment_factor = 1+round(team_2_off_pct + ((team_1_def_eff/def_eff_avg) - 1),2)
    team_2_off_adj = round(off_eff_avg * team_2_off_pct_adjustment_factor,2)
    if verbose:
        if neutral:
            print(f"{team_1}'s expected efficiency is {team_1_off_adj}")
            print(f"{team_2}'s expected efficiency is {team_2_off_adj}")
            print(f"{team_1} averages {round(team_1_tempo)} possessions per game")
            print(f"{team_2} averages {round(team_2_tempo)} possessions per game")
            print(f"{team_1} is therefore expected to score {round((team_1_tempo)*(team_1_off_adj/100))} points")
            print(f"{team_2} is therefore expected to score {round((team_2_tempo)*(team_2_off_adj/100))} points")
        else:
            print(f"{team_1}'s expected efficiency is {team_1_off_adj}")
            print(f"{team_2}'s expected efficiency is {team_2_off_adj}")
            print(f"{team_1} averages {round(team_1_tempo)} possessions per game")
            print(f"{team_2} averages {round(team_2_tempo)} possessions per game")
            print(f"{team_1} is therefore expected to score {round((team_1_tempo)*(team_1_off_adj/100)+3.75)} points")
            print(f"{team_2} is therefore expected to score {round((team_2_tempo)*(team_2_off_adj/100))} points")
    if neutral:
        return(round((team_1_tempo)*(team_1_off_adj/100)),round((team_2_tempo)*(team_2_off_adj/100)))
    else:
        return(round((team_1_tempo)*(team_1_off_adj/100)+3.75),round((team_2_tempo)*(team_2_off_adj/100)))

"""
To do list:
 - Add the visual option to spit out a shap value waterfall plot for the prediction.  Will need to map colors per school
    - This will also require pickling the shap.explainer object for the w_l model, which will require re-training
 - Clean up output of the dataframe function
- Try training the models without the simple rating system - it may be a redundant feature.
- Try training the models with the KenPom four factors data.
"""