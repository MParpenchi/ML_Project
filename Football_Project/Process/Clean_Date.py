import pandas as pd

# Dictionary definition

event_types = {0:'Announcement', 1:'Attempt', 2:'Corner', 3:'Foul', 4:'Yellow card', 5:'Second yellow card', 6:'Red card', 7:'Substitution', 8:'Free kick won', 9:'Offside', 10:'Hand ball', 11:'Penalty conceded'}
event_types2 = {12:'Key Pass', 13:'Failed through ball', 14:'Sending off', 15:'Own goal'}
sides = {1:'Home', 2:'Away'}
shot_places = {1:'Bit too high', 2:'Blocked', 3:'Bottom left corner', 4:'Bottom right corner', 5:'Centre of the goal', 6:'High and wide', 7:'Hits the bar', 8:'Misses to the left', 9:'Misses to the right', 10:'Too high', 11:'Top centre of the goal', 12:'Top left corner', 13:'Top right corner'}
shot_outcomes = {1:'On target', 2:'Off target', 3:'Blocked', 4:'Hit the bar'}
locations = {1:'Attacking half', 2:'Defensive half', 3:'Centre of the box', 4:'Left wing', 5:'Right wing', 6:'Difficult angle and long range', 7:'Difficult angle on the left', 8:'Difficult angle on the right', 9:'Left side of the box', 10:'Left side of the six yard box', 11:'Right side of the box', 12:'Right side of the six yard box', 13:'Very close range', 14:'Penalty spot', 15:'Outside the box', 16:'Long range', 17:'More than 35 yards', 18:'More than 40 yards', 19:'Not recorded'}
bodyparts = {1:'Right foot', 2:'Left foot', 3:'Head'}
assist_methods = {0:'None', 1:'Pass', 2:'Cross', 3:'Headed pass', 4:'Through ball'}
situations = {1:'Open play', 2:'Set piece', 3:'Corner', 4:'Free kick'}

# Data loading

print("Loading data...")
events = pd.read_csv('events.csv')
ginf = pd.read_csv('ginf.csv')

# merging data

print("Merging data...")
match_metadata = ginf[['id_odsp', 'date', 'league', 'season', 'country']]
df = pd.merge(events, match_metadata, on='id_odsp', how='left')

# mapping codes to strings
# look in the event_type column, for every number you see, take its text equivalent from the dictionary and write it into a new column called event_type_str."
print("Mapping codes to text...")
df['event_type_str'] = df['event_type'].map(event_types)
df['event_type2_str'] = df['event_type2'].map(event_types2)
df['side_str'] = df['side'].map(sides)
df['shot_place_str'] = df['shot_place'].map(shot_places)
df['shot_outcome_str'] = df['shot_outcome'].map(shot_outcomes)
df['location_str'] = df['location'].map(locations)
df['bodypart_str'] = df['bodypart'].map(bodyparts)
df['assist_method_str'] = df['assist_method'].map(assist_methods)
df['situation_str'] = df['situation'].map(situations)

# Filling null values. In soccer, many goals do not have an assist (the player dribbled and scored). The database has an empty (NaN) placeholder for the assist. If we don't fix this, the model will think the data is corrupt. We will write Solo Run instead. If the player name is empty, we will write Unknown to avoid an error.

df['assist_method_str'] = df['assist_method_str'].fillna('Solo Run') 
df['player'] = df['player'].fillna('Unknown')

# ---------------------------------------------------------
# Classification
# ---------------------------------------------------------
#Keep only rows where event_type is 1. Why? According to the dictionary, the number 1 means Attempt.
#For the "Goal Prediction" project, we don't need yellow cards, substitutions, or handballs. We just want the moments when a shot is taken to see if it was scored. This line reduces the data size and increases the accuracy of the model.
print("Filtering for shots...")
shots_df = df[df['event_type'] == 1].copy()

#choosing c=important columns
final_columns = [
    'id_odsp', 'id_event', 'sort_order', 'time', # ID and time
    'league', 'season', 'date', 'country',       # Game metadata
    'event_team', 'opponent', 'player',          # Teams and players
    'is_goal',                                   # (Target Variable)
    'shot_place_str', 'shot_outcome_str',        # Shot Features
    'location_str', 'bodypart_str',              # Features of place and body
    'assist_method_str', 'situation_str',        # Pass and play conditions
    'fast_break'                                 # Was it a counterattack? 
]

cleaned_data = shots_df[final_columns]

# ---------------------------------------------------------
# Saving file
# ---------------------------------------------------------
output_filename = 'cleaned_shots_data.csv'
cleaned_data.to_csv(output_filename, index=False)
print(f"Done! Cleaned data saved as '{output_filename}'")
print(f"Total shots extracted: {len(cleaned_data)}")