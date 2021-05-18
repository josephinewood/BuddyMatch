import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor

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
    nbrs = NearestNeighbors(n_neighbors = 4, metric = 'dice', algorithm = 'brute').fit(lfb) # train on the people looking for buddies
    distances, indices = nbrs.kneighbors(buds) # match buddies to best fit
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
            lfb_mbr = buddies.iloc[lfbs[j]]['Name']
            bud_mbr = buddies.iloc[bud]['Name']
            dist = distances[i][j]
            d = pd.DataFrame([[bud_mbr, lfb_mbr, dist]], columns = ['Buddy', 'New Member', 'Distance'])
            pairs_df = pairs_df.append(d)
    return(pairs_df)

def find_pairs(buddies, n_neighbors = 4):
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
    buddies['Interest 1'] = buddies['Interest 1']*3
    buddies['Interest 2'] = buddies['Interest 2']*2
    buddies['Interest 3'] = buddies['Interest 3']*1
    
    # remove person ids, their names won't give us any information (and will mess with the NN alg)
    lfb = buddies[buddies['Buddy'] == 0][['Name', 'Interest 1', 'Interest 2', 'Interest 3', 'Neighborhood', 'Age']]
    buds = buddies[buddies['Buddy'] == 1][['Name', 'Interest 1', 'Interest 2', 'Interest 3', 'Neighborhood', 'Age']]
    
    # Fit NN and consolidate matches into dataframe
    NN = mbr_NN(lfb, buds, n_neighbors = 4)
    pairs_df = pairs_dataframe(lfb, buds, buddies, NN)
    return(pairs_df)