import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor

# This is pretty hazy stats wise, but I need a way to map categorical  
# variables to numeric and have some sort of 'closeness' between them

interests_dict = {'Fitness': 1,
                 'Sports': 2,
                 'Outdoors/Hiking': 3,
                 'Exploring the city': 4,
                 'Cooking': 5,
                 'Movies/TV': 6,
                 'Music': 7,
                 'Reading': 8,
                 'Arts & Crafts': 9
}
neighborhoods = ['River North','Old Town','Gold Coast','Streeterville','Loop',
                 'Lincoln Park','Lakeview','Wrigley','West Loop','Uptown','Edgewater',
                 'Rogers Park','Wicker Park','Bucktown','Logans Square','South Loop',
                 'Near South Side','Hyde Park','Pilsen']

neighborhoods_dict = {'River North': 1,
                 'Old Town': 1,
                 'Gold Coast': 1,
                 'Streeterville': 1,
                 'Loop': 1,                 # these are based on proximity to the loop
                 'Lincoln Park': 2,         # should do miles but idk where to get that info
                 'Lakeview': 2,
                 'Wrigley': 2,
                 'West Loop': 2,
                 'Uptown': 3,
                 'Edgewater': 3,
                 'Rogers Park': 3,
                 'Wicker Park': 3,
                 'Bucktown': 3,
                 'Logans Square': 3,
                 'South Loop': 5,
                 'Near South Side': 5,
                 'Hyde Park': 5,
                 'Pilsen': 5,
                 'West Loop, Wicker, Bucktown, or Logan Square': 3
}

def map_interests(data, interests_map, n_interests = 3):
    # map interests
    cols = []
    for i in range(1, n_interests + 1):
        colname = 'Interest' + str(i)
        data[colname] = data[colname].replace(interests_map)
        cols.append(colname)
    new = data[cols]
    return(new)

def map_neighborhoods(data, neighborhoods_map):
    # map neighborhoods
    cols = []
    for n in range(0, len(neighborhoods_map)):
        colname = 'Neighborhood' + str(n)
        if colname in data.columns:
            data[colname] = data[colname].replace(neighborhoods_map)
            cols.append(colname)
        else:
            pass
    new = data[cols]
    return(new)

def map_categoricals(data, neighborhoods_map, interests_map, n_interests = 3):
    interests = map_interests(data, interests_map, n_interests)
    neighborhoods = map_neighborhoods(data, neighborhoods_map)
    new = data[['Name', 'Age', 'Buddy']].join(interests).join(neighborhoods)
    return(new)

def mbr_NN(lfb, buds, n_neighbors = 4):
    '''
    Fits a NearestNeighbors model to the new members, and then
    calls it on the current members to obtain their match.
    lfb - new member dataframe
    buds - current member dataframe
    n_neighbors - how many neighbors to find, 
                  I chose 4 to have a higher 
                  chance of finding a match
    '''
    # all neighborhood columns
    neighborhoods_cols = [col for col in lfb.columns if 'Neighborhood' in col]

    nbrs = NearestNeighbors(n_neighbors = 4, metric = 'dice', algorithm = 'brute').fit(lfb[['Age', 'Interest1', 'Interest2', 'Interest3'] + neighborhoods_cols]) # train on the people looking for buddies
    distances, indices = nbrs.kneighbors(buds[['Age', 'Interest1', 'Interest2', 'Interest3'] + neighborhoods_cols]) # match buddies to best fit
    return(distances, indices)

def pairs_dataframe(lfb, buds, buddies, NN):
    '''
    Results in a dataframe of current members and all of 
    their new member matches, plus the distance between them.
    lfb - new member dataframe
    buds - current member dataframe
    NN - NearestNeighbors, nbrs.kneighbors from the mbr_NN function
    '''
#     pairs[i] are the buddies
#     pairs[i][1] is a list of the lfb the buddy was paired with
#     pairs[i][1][j] is one paired lfb

    distances, indices = NN
    lfb_idx = lfb.index
    buds_idx = buds.index
    pairs = list(zip(buds_idx, indices))
    pairs_df = pd.DataFrame([])
    for i in range(0, len(pairs) - 1):
        lfbs = pairs[i][1]
        bud = pairs[i][0]
        for j in range(0, len(lfbs)): # for lfb mbr in matches of pairs[p][0]
            lfb_mbr = lfb.iloc[lfbs[j]]['Name']
            bud_mbr = buds.iloc[j]['Name']
            dist = distances[i][j]
            d = pd.DataFrame([[bud_mbr, lfb_mbr, dist]], columns = ['Buddy', 'New Member', 'Distance'])
            pairs_df = pairs_df.append(d)
    return(pairs_df)

def find_pairs(data, n_neighbors = 4, all_buddies = False):
    '''
    Runs entire data preprocessing and NearestNeighbors 
    process, returns the pairs dataframe.
    lfb - new member dataframe
    buds - current member dataframe
    NN - NearestNeighbors, nbrs.kneighbors from the mbr_NN function
    n_neighbors - how many neighbors to find, 
                  I chose 4 to have a higher 
                  chance of finding a match
    '''
    # weight the ranked interests to reflect importance
    data['Interest1'] = data['Interest1']*3
    data['Interest2'] = data['Interest2']*2
    data['Interest3'] = data['Interest3']*1
    
    # all neighborhood columns
    neighborhoods_cols = [col for col in data.columns if 'Neighborhood' in col]
    
    # remove person ids, their names won't give us any information (and will mess with the NN alg)
    lfb = data[data['Buddy'] == 1]
    buds = data[data['Buddy'] == 0]
    
    # Fit NN and consolidate matches into dataframe
    NN = mbr_NN(lfb[['Age', 'Interest1', 'Interest2', 'Interest3'] + neighborhoods_cols], buds[['Name', 'Age', 'Interest1', 'Interest2', 'Interest3'] + neighborhoods_cols], n_neighbors = 4)
    pairs_df = pairs_dataframe(lfb, buds, data, NN).sort_values(['New Member', 'Distance'], ascending = False)
    if all_buddies == True:
        return(pairs_df)
    else:
        pairs_df = pairs_df.drop_duplicates(['Buddy', 'New Member'])
        return(pairs_df)