print(‘>> Installing Libraries’)
!pip3 install pandas matplotlib numpy scikit-surprise
print(‘>> Libraries Installed’)
print(‘>> Importing Libraries’)
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.accuracy import rmse, mae
from surprise.model_selection import cross_validate
print(‘>> Libraries imported.’)
df=pd.read_csv(‘ratings.csv’)
df.head()
df.drop(‘timestamp’, axis=1, inplace=True)
df.head()
df.isna().sum()
n_movies=df[“movieId”].nunique()
n_users=df[“userId”].nunique()
print(f’Number of unique movies: {n_movies}’)
print(f’Number of unique users: {n_users}’)
available_ratings=df[‘rating’].count()
25
total_ratings=n_movies*n_users
missing_ratings=total_ratings – available_ratings
sparsity=(missing_ratings/total_ratings)*100
print(f’Sparsity: {sparsity}’)
df[‘rating’].value_counts().plot(kind= ‘bar’)
filter_movies=df[‘movieId’].value_counts() > 3
filter_movies=filter_movies[filter_movies].index.tolist()
filter_users=df[‘userId’].value_counts() > 3
filter_users=filter_users[filter_users].index.tolist()
print(f’Original Shape: {df.shape}’)
df=df[(df[‘movieId’].isin(filter_movies)) & (df[‘userId’].isin(filter_users))]
print(f’New shape: {df.shape}’)
cols = [‘userId’, ‘movieId’, ‘rating’]
reader = Reader(rating_scale = (0.5 , 5))
data = Dataset.load_from_df(df[cols], reader)
trainset = data.build_full_trainset()
antiset = trainset.build_anti_testset()
algo = SVD(n_epochs = 25, verbose= True)
cross_validate(algo, data, measures) = [‘RMSE’, ‘MAE’], cv=5, verbose = True)
print(‘>> Training Done’)
predictions = algo.test(antiset)
predictions[0]
from collections import defaultdict
def get_top_n(predictions, n):
top_n = defaultdict(list)
for uid, iid, _, est, _ in predictions:
26
top_n[uid].append((iid, est))
for uid, user_ratings in top_n.items():
user_ratings.sort(key = lambda x: x[1], reverse = True)
top_n[uid] = user_ratings[:n]
return top_n
pass
top_n = get_top_n(predictions, n=3)
for uid, user_ratings in top_n.items():
print(uid, [iid for (iid, rating) in user_ratings])
