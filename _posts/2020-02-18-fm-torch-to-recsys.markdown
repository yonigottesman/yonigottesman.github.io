---
layout: post
title:  "Movie Recommender from Pytorch to Elasticsearch"
excerpt: "Build and serve a movie recommender from scratch using movielens, pytorch, factorization machines and elasticsearch "
date:   2020-02-18 06:48:38 +0200
categories: recsys pytorch elasticsearch
---

<!-- Mathjax Support -->
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


In this post I'll build and serve a production grade movie recommender from scratch!  :open_mouth:  
I'll use the movielens 1M dataset to train a Factorization Machine model implemented with pytroch. After learning the vector representation of movies and user metadata I'll use elasticsearch to serve the model and recommend movies to new users.  
The full code is here: [github](https://github.com/yonigottesman/recommendation_playground/blob/master/fatorization_machine-1M.ipynb), [colab](https://colab.research.google.com/drive/1z21Hb-CORQ0JbH7cPSwULcY-JhhlaeQe)



(A short) Introduction to Factorization Machines
----------------------------------------------
[Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)  (FM) is a supervized machine learing model that extends traditional matrix factorization by also learning interactions between different feature values of the model. Another advantage of FM is that it solves the cold start problem, we can make predictions based on user metadata (age, gender etc.) even if its a user the system has never seen before.  

The input to the model is a list of events that looks like this:

![FM feature matrix]({{ "/assets/fm_feature_vector.png" | absolute_url }})

Each row is an event, for example $$ x^{(1)} $$ is the event that user A rated movie TI at time 13 with the value $$y^{(1)}=5$$. The feature values are the different types of columns we have, for example for users we have A, B, C etc. feature values. The number of feature values is the width of a row.  
In this case there is some more metadata like "Last Movie rated", but in our movielens dataset we will add user age, gender and occupation instead.  


The model equation for a factorization machine of degree d=2 is defined as:  

$$
\begin{align*}
 & \hat y(x) := w_0 + \sum_{i = 1}^n w_ix_i + \sum_{i = 1}^n\sum_{j = i + 1}^{n}<v_i, v_j>x_ix_j
\end{align*}
$$


Each feature value (column in the input matrix) will be assiged an embeddings vector $$v_i$$ of size $$k$$ and a bias factor $$w_i$$ that will be learned during the training process. $$<v_i,v_j>$$ models the interaction between the $$i$$-th and $$j$$-th feature value. Hopefully the embeddings will capture some interesting semantics and similar movies/users will have similar embeddings.


Pytorch FM Model
----------------

Without further ado, lets see the pytorch implementation of the formula above:

```python
class FMModel(nn.Module):
  def __init__(self, n, k):
    super().__init__()

    self.w0 = nn.Parameter(torch.zeros(1))
    self.bias = nn.Embedding(n, 1)
    self.embeddings = nn.Embedding(n, k)

    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad(): trunc_normal_(self.embeddings.weight, std=0.01)
    with torch.no_grad(): trunc_normal_(self.bias.weight, std=0.01)

  def forward(self, X):
    emb = self.embeddings(X)
    # calculate the interactions in complexity of O(nk) see lemma 3.1 from paper
    pow_of_sum = emb.sum(dim=1).pow(2)
    sum_of_pow = emb.pow(2).sum(dim=1)
    pairwise = (pow_of_sum-sum_of_pow).sum(1)*0.5
    bias = self.bias(X).squeeze().sum(1)
    return torch.sigmoid(self.w0 + bias + pairwise)*5.5
```

There are a few things to point out here:
   * I'm implementing lamma 3.1 from the paper that prooves that 
   $$ 
   \sum_{i = 1}^n\sum_{j = i + 1}^{n}<v_i, v_j>x_ix_j =  \sum_{f=1}^k((\sum_{i=1}^{n}v_{i,f}x_i)^2-(\sum_{i=1}^{n}v_{i,f}^2x_i^2))
   $$  
   This means the pairwise interaction are done in $$O(kn)$$ and not $$O(kn^2)$$  
   
   * I'm using an Embeddings layer, so the input will be a tensor of offsets and not a hot-encoded vector (like in the image). So $$x_i$$ from before can be $$[0,1,0,0,1,1,0]$$ but the input to the model will be $$[1,4,5]$$  
   
   * I wrap the result with a sigmoid function to limit $$\hat y(x)$$ to be between 0 and 5.5. This helps the model to learn mutch faster.
   
   * The embeddings are initialized with a truncated normal function - This is something I've learned from the [fastai](https://github.com/fastai/fastai2/blob/master/fastai2/layers.py#L346) library and improves learning spead alot.

Prepare and Fit Data
----------------
First download the 1M movielens dataset from [here](http://files.grouplens.org/datasets/movielens/ml-1m.zip).
After reading and chewing the data I'm left with an event input matrix like before (see code for more details):

|    |   userId |   movieId |   age |   gender |   occupation |   rating |
|----|------------------|-------------------|---------------|------------------|----------------------|----------|
|  0 |                0 |              7216 |          9746 |             9753 |                 9765 |        5 |
|  1 |                0 |              6695 |          9746 |             9753 |                 9765 |        3 |
|  2 |                0 |              6942 |          9746 |             9753 |                 9765 |        3 |
|  3 |                0 |              9379 |          9746 |             9753 |                 9765 |        4 |


The numbers in the matrix represent the feature value index. I could transform each row to a hot-encoded vector like in the paper but im using pytorch Embeddings layer that expects a list of indices.

Next step is to split the data to train and validation and create pytorch dataloader:
```python
data_x = torch.tensor(ratings[feature_columns].values)
data_y = torch.tensor(ratings['rating'].values).float()
dataset = data.TensorDataset(data_x, data_y)
bs=1024
train_n = int(len(dataset)*0.9)
valid_n = len(dataset) - train_n
splits = [train_n,valid_n]
assert sum(splits) == len(dataset)
trainset,devset = torch.utils.data.random_split(dataset,splits)
train_dataloader = data.DataLoader(trainset,batch_size=bs,shuffle=True)
dev_dataloader = data.DataLoader(devset,batch_size=bs,shuffle=True)
```

Standard fit/test single epoch loops:
```python
def fit(iterator, model, optimizer, criterion):
    train_loss = 0
    model.train()
    for x,y in iterator:
        optimizer.zero_grad()
        y_hat = model(x.to(device))
        loss = criterion(y_hat, y.to(device))
        train_loss += loss.item()*x.shape[0]
        loss.backward()
        optimizer.step()
    return train_loss / len(iterator.dataset)

def test(iterator, model, criterion):
    train_loss = 0
    model.eval()
    for x,y in iterator:                    
        with torch.no_grad():
            y_hat = model(x.to(device))
        loss = criterion(y_hat, y.to(device))
        train_loss += loss.item()*x.shape[0]
    return train_loss / len(iterator.dataset)
```
Create the model with $$k=120$$ (length of each feature value embedding), and train:

```python
model = FMModel(data_x.max()+1, 120).to(device)
wd=1e-5
lr=0.001
epochs=10
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7], gamma=0.1)
criterion = nn.MSELoss().to(device)
for epoch in range(epochs):
  start_time = time.time()
  train_loss = fit(train_dataloader, model, optimizer, criterion)
  valid_loss = test(dev_dataloader, model, criterion)
  scheduler.step()
  secs = int(time.time() - start_time)
  print(f'epoch {epoch}. time: {secs}[s]')
  print(f'\ttrain rmse: {(math.sqrt(train_loss)):.4f}')
  print(f'\tvalidation rmse: {(math.sqrt(valid_loss)):.4f}')
```
After 10 epochs the rmse is **0.8532**, from what I've seen on various leaderboards this is fine for a vanilla FM model.


Understand Embeddings
----------------
After the model finished training, each feature value has its own embedding and hopyfully the model managed to capture some semantics just like word2vec does for words.
I expect similar feature values, for example similar movies or users with similar taste, to be close to one another when measuring distance with cosine similarity.  

Ill visualize embeddings of movies in these different genres: Children\'s, Horror and Documentary on a 2D plot using t-SNE for dimensionality reduction:


```python
movies_subset = movies[movies['genres'].str.contains('Children\'s|Horror|Documentary')].copy()
X = np.stack(movies_subset['embedding'].values)
ldr = TSNE(n_components=2, random_state=0)
Y = ldr.fit_transform(X)
movies_subset['x'] = Y[:, 0]
movies_subset['y'] = Y[:, 1]
ax = sns.scatterplot(x="x", y="y", hue='genres',data=movies_subset)
```

![movie embeddings]({{ "/assets/movie_emb.jpeg" | absolute_url }})

Cool! the model managed to seperate genres without me telling it the real genre!

Make Movie Recomendations
----------------
We can now make movie recommendations to users we have seen **and** users we have not seen that we only have some metadata about (age, gender or occupation).
All we have to do is calculate the FM equations once for each movie and see which movies get the highest predicted rating.  
For example if we have a new (never seen in train so no userId embedding) male user (male_index=9754) and his age is between 18 and 25 (age18_25_index=9747) we will calculate $$\hat y(x)$$ once for every movie:  

$$ 
\hat y(9754,9747,movie_i):= w_0 + w_{9747}+ w_{9754} + w_{movie_i}+ \sum_{i = \{9747,9754,movie_i\}}\sum_j <v_i, v_j>x_ix_j
$$

And choose the $$ movie_i$$'s with the highest ratings. **But** the nice thing here is that for recommendation I dont need the rating itself, its enough to know which movies were best without calculating the full equation.
I can do this by ignoring all the terms in $$ \hat y(x)$$ that are the same for each movie calculation, and then I'm left with:

$$
score(movie_i):= w_{movie_i} +<v_{movie_i},v_{9754}> + <v_{movie_i},v_{9747}>
$$

which is the same as 

$$
score(movie_i):= w_{movie_i} + <(v_{9754}+v_{9747}),v_{movie_i}>
$$

To summerize, when a user with metadata x,y,z arrives and needs a recommendation, I will sum the metadata $$ v_{metadata}=v_x+v_y+v_z$$ and rank all the movies by cacluating:

$$ rank(movie_i):=w_{movie_i}+<v_{metadata},movie_i>$$

Lets recommend top 10 movies to a male aged 18 to 25 unknown user with these steps:
1. Get "man" embeddigns $$v_{9754}$$
2. Get "age 18 to 25" embeddings $$v_{9747}$$
3. Calculate $$v_{metadata}=v_{9754} + v_{9747}$$
4. Calculate rank for each movie and return top 10

And for the code:
```python
man_embeddings = model.embeddings(torch.tensor(9754,device=device))
age18_25_embeddings = model.embeddings(torch.tensor(9747,device=device))
metadata_embeddings = man_embeddings+age18_25_embeddings
ranknings = movie_bias.squeeze()+(metadata_embeddings*movie_embeddings).sum(1)
movies.iloc[ranknings.argsort(descending=True)]['title'].values[:10]
``` 

```
['Shawshank Redemption, The (1994)',
 'Usual Suspects, The (1995)',
 'American Beauty (1999)',
 'Godfather, The (1972)',
 'Life Is Beautiful (La Vita è bella) (1997)',
 'Braveheart (1995)',
 'Monty Python and the Holy Grail (1974)',
 "Schindler's List (1993)",
 'GoodFellas (1990)',
 'Star Wars: Episode IV - A New Hope (1977)']
``` 

And now lets recommend top 10 movies to a female over 56 unknown user:
```python
woman_embeddings = model.embeddings(torch.tensor(9753,device=device))
over56_embeddings = model.embeddings(torch.tensor(9752,device=device))
metadata_embeddings = woman_embeddings+over56_embeddings
ranknings = movie_bias.squeeze()+(metadata_embeddings*movie_embeddings).sum(1)
movies.iloc[ranknings.argsort(descending=True)]['title'].values[:10]
``` 

```
['Gone with the Wind (1939)',
'To Kill a Mockingbird (1962)',
"Schindler's List (1993)",
'Some Like It Hot (1959)',
'African Queen, The (1951)',
'Color Purple, The (1985)',
'Grapes of Wrath, The (1940)',
'Room with a View, A (1986)',
'Casablanca (1942)',
'Wizard of Oz, The (1939)']
``` 

Seems about right ¯\\_(ツ)_/¯  

Serve Recommender System with Elasticsearch
----------------
The last step in building my recommender is to move from my jupyter code to a realltime serving system.
I will use elasticsearch to store all my feature value embeddings $$v_i$$ and biases $$w_i$$.
This is how the recommendation flow will go:
1. User enters my system (netflix website for example)
2. Query elasticsearch for user metadata vectors and add them to create $$v_{metadata}$$
3. Query elasticsearch again, this time sending $$v_{metadata}$$ and ranking movies by $$ rank(movie_i):=w_{movie_i}+<v_{metadata},movie_i>$$
4. Display user the results

First thing is to get elasticseach running, creating an index and indexing all embeddings.
Lets [quickstart](https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started-install.html) by downloading and running elasticsearch on a single node with default configurations:

```shell
curl -L -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.0-darwin-x86_64.tar.gz
tar -xvf elasticsearch-7.6.0-darwin-x86_64.tar.gz
cd elasticsearch-7.6.0/bin
./elasticsearch
```

I will now create a new index in my cluster, each document will have 4 fields, "embeddings" of type "dense_vector" this type will allow us to perform the dot product operation, "bias" is $$w_i$$, "feature_type" is the type of feature this doc represents (user, movie, user_metadata) and will be used for filtering, title is the name of the movie.

```shell
curl -X PUT "localhost:9200/recsys?pretty" -H 'Content-Type: application/json' -d'
{
  "mappings": {
    "properties": {
      "feature_type":{
        "type":"keyword"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 120
      },
      "bias": {
        "type":"double"
      },
      "title" : {
        "type" : "keyword"
      }
    }
  }
}
'
```

Back to my python notebook, I'll now create a document from each movie and index it:

```python
def generate_movie_docs():
    for i, movie in movies.iterrows():
        yield {
            '_index': 'recsys',
            '_id': f'movie_{movie["movieId"]}',
            'doc': {'embedding':movie['embedding'].tolist(),
                    'bias':movie['bias'],
                    'feater_type':'movie',
                    'title':movie['title']
                   }
        }
es = Elasticsearch()
helpers.bulk(es,generate_movie_docs())
```

Ill do the same for users, age vectors, gender vectors and occupation. The only difference between them is 'feature_type' which I use later to filtering.  
Ill still need to store somewhere the mapping between feature value to index (for example female index is 9753), these indices where chosen when I created the datasets. I can store them anywhere and load them to memory when the systems starts.  

Now everything is set, we can start making predictions according to the flow:
1. The same unknown user from before - male between 18-25 enters my website!
2. Query elasticsearch for user metadata vectors and add them to create $$v_{metadata}$$

```python
metadata = es.mget({"docs":[
    {
        "_index" : "recsys",
        "_id" : "age_9747"},
    {
        "_index" : "recsys",
        "_id" : "gender_9754"}]})
embeddings = [doc['_source']['embedding'] for doc in metadata['docs']]
v_metadata = [sum(pair) for pair in zip(*embeddings)]
```



3. Query elasticsearch again, this time sending $$v_{metadata}$$ and ranking movies by $$ rank(movie_i):=w_{movie_i}+<v_{metadata},movie_i>$$

```python
search_body = {
    "query": {
    "script_score": {
      "query" : {
          "bool" : {
          "filter" : {
            "term" : {
              "feature_type" : "movie" 
            }
          }
        }
      },
      "script": {
        "source": "dotProduct(params.query_vector, \u0027embedding\u0027) + doc[\u0027bias\u0027].value", 
        "params": {
          "query_vector": v_metadata
        }
      }
    }
  }
}
[hit['_source']['title'] 
 for hit in es.search(search_body,index='recsys',_source_includes='title')['hits']['hits']]
```

* The filter part is for only calculating movie documents.
* The "source" script is the ranking method" on each movie document with the v_metadata from stage 2.

And the results...
```
['Shawshank Redemption, The (1994)',
 'Usual Suspects, The (1995)',
 'American Beauty (1999)',
 'Godfather, The (1972)',
 'Life Is Beautiful (La Vita è bella) (1997)',
 'Braveheart (1995)',
 'Monty Python and the Holy Grail (1974)',
 'Star Wars: Episode IV - A New Hope (1977)',
 'Sanjuro (1962)',
 'GoodFellas (1990)']
```

Awsome! we are out from a jupyter playground to a real world, production quality serving system!
