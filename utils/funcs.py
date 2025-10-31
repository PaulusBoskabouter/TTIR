import pandas as pd 

def fetch_user_likes(timestamp:int, user:int, likes:pd.DataFrame, unlikes:pd.DataFrame, dislikes:pd.DataFrame, undislikes:pd.DataFrame) -> dict:
    like_tracker = {}
    
    # Subset our dataframes to contain only user id and timestramp up until the given timestamp parameter.
    likes = likes[(likes['uid'] == user) & (likes['timestamp'] <= timestamp)]
    unlikes = unlikes[(unlikes['uid'] == user) & (unlikes['timestamp'] <= timestamp)]
    dislikes = dislikes[(dislikes['uid'] == user) & (dislikes['timestamp'] <= timestamp)]
    undislikes = undislikes[(undislikes['uid'] == user) & (undislikes['timestamp'] <= timestamp)]

    if len(likes) == 0 and len(dislikes) == 0:
        print('No user interactions found')
        return {}
    
    print(len(likes), len(dislikes))
    # For likes
    for row in range(len(likes)):
        song = likes.iloc[row]['item_id']
        try:
            like_tracker[song] += 1
        except KeyError:
            like_tracker[song] = 1
    
    # For dislikes
    for row in range(len(dislikes)):
        song = dislikes.iloc[row]['item_id']
        try:
            like_tracker[song] -= 1
        except KeyError:
            like_tracker[song] = -1
    
    

    # For these two we make no exceptions, it should not be that there's songs in here which likes/dislikes haven't added before so let it crash
    # For unlikes 
    for row in range(len(unlikes)):
        song = unlikes.iloc[row]['item_id']
        like_tracker[song] -= 1
    
    # For undislikes
    for row in range(len(undislikes)):
        song = undislikes.iloc[row]['item_id']
        like_tracker[song] += 1

            
    return like_tracker