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


In this post ill build and serve a movie recommender from scratch!  :open_mouth:  
I'll use the movielens 1M dataset to train a Factorization Machine model implemented with pytroch. After learning the vector representation of movies and user metadata I'll use elasticsearch to serve the model and recommend movies to new users.  
The full code is here: [github](https://github.com/yonigottesman/recommendation_playground/blob/master/fatorization_machine-1M.ipynb), [colab](https://colab.research.google.com/drive/1z21Hb-CORQ0JbH7cPSwULcY-JhhlaeQe)


(A short) Introduction to Factorization Machines
----------------------------------------------
[Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)  (FM) is a supervized machine learing model that extends traditional matrix factorization by also learning interactions between different feature values of the model. Another advantage of FM is that it solves the cold start problem, we can make predictions based on user metadata (age, gender etc.) even if its a user the system has never seen before.  

The input to the model is a list of events that looks like this:

![FM feature matrix]({{ "/assets/fm_feature_vector.png" | absolute_url }})

Each row is an event, for example $$ x^{(1)} $$ 
is just the event that user A rated movie TI at time 13 with the value $$y^{(1)}=5$$. The feature values are the different types of columns we have, for example for users we have A, B, C etc. feature values. The number of feature values is the width of a row.  
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

{% highlight python %}
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
{% endhighlight %}

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
{% highlight python %}
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
{% endhighlight %}

Standard fit/test single epoch loops:
{% highlight python %}
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
{% endhighlight %}

Create the model with $$k=120$$ (length of each feature value embedding), and train:
{% highlight python %}
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
{% endhighlight %}

After 10 epochs the rmse is **0.8532**, from what i've seen on various leaderboards this seems ok for a vanilla FM model.


Understand Embeddings
----------------
After the model finished training, each feature value has its own embedding and hopyfully the model managed to capture some semantics just like word2vec does for words.
I expect similar feature values, for example similar movies or users with similar taste, to be close to one another when measuring distance with cosine similarity.  

Ill visualize embeddings of movies in these different genres: Children\'s, Horror and Documentary, on a 2D plot using t-SNE for dimensionality reduction:


{% highlight python %}
unique_movies = (ratings[ratings['genres']
                         .str.contains('Children\'s|Horror|Documentary')]
                 .drop_duplicates('movieId_indices').copy())
X = model.embeddings(torch.tensor(unique_movies['movieId_indices']
                                  .values,device=device)).cpu().detach()
ldr = TSNE(n_components=2, random_state=0)
Y = ldr.fit_transform(X)
unique_movies['x'] = Y[:, 0]
unique_movies['y'] = Y[:, 1]
ax = sns.scatterplot(x="x", y="y", hue='genres',data=unique_movies)
{% endhighlight %}

![movie embeddings]({{ "/assets/movie_emb.jpeg" | absolute_url }})

Cool! the model managed to seperate genres without me telling it the real genre!

Make Movie Recomendations
----------------
We can now make movie recommendations to users we have seen **and** users we have not seen that we only have some metadata about (age, gender or occupation).
All we have to do is calculate the FM equations once for each movie and see which movies get the highest predicted rating.  
For example if we have a new male user (male_index=9754) and his age is under 18 (under18_index=9746) we will calculate $$\hat y(x)$$ once for every movie:  

$$ 
\hat y(9754,9746,movie_i):= w_0 + w_{9746}+ w_{9754} + w_{movie_i}+ \sum_{i = \{9746,9754,movie_i\}}\sum_j <v_i, v_j>x_ix_j
$$

And choose the $$ movie_i$$'s with the highest ratings. **But** the nice thing here is that for recommendation I dont need the rating itself, its enough to know which movies were best without calculating the full equation.
I can do this by ignoring all the terms in $$ \hat y(x)$$ that are the same for each movie calculation, and then I'm left with:

$$
score(movie_i):= w_{movie_i} +<v_{movie_i},v_{9754}> + <v_{movie_i},v_{9746}>
$$

wich is the same as 

$$
score(movie_i):= w_{movie_i} + <(v_{9754}+v_{974}),v_{movie_i}>
$$

To summerize, when a user with metadata x,y,z arrives and needs a recommendation, I will sum the metadata $$ v_{metadata}=v_x+v_y+v_z$$ and rank all the movies by cacluating:

$$ rank(movie_i):=w_{movie_i}+<v_{metadata},movie_i>$$

Lets recommend top 20 movies to a male under 18 unknown user:
{% highlight python %}
man_embeddings = model.embeddings(torch.tensor(9754,device=device))
under18_embeddings = model.embeddings(torch.tensor(9746,device=device))
metadata_embeddings = man_embeddings+under18_embeddings
dot_results = movie_bias.squeeze()+(metadata_embeddings*movie_embeddings).sum(1)
[index_to_title[movies_tensor[i.item()].item()] 
 for i in dot_results.argsort(descending=True)][:20]w_{movie_i}
{% endhighlight %} 

{% highlight python %}
['GoodFellas (1990)',
'Shawshank Redemption, The (1994)',
'Godfather, The (1972)',
'Star Wars: Episode IV - A New Hope (1977)',
'Usual Suspects, The (1995)',
'Raiders of the Lost Ark (1981)',
'Matrix, The (1999)',
'Pulp Fiction (1994)',
'American Beauty (1999)',
'Apocalypse Now (1979)',
'Princess Mononoke, The (Mononoke Hime) (1997)',
'Forrest Gump (1994)',
"One Flew Over the Cuckoo's Nest (1975)",
'Saving Private Ryan (1998)',
'Monty Python and the Holy Grail (1974)',
'Animal House (1978)',
'Patton (1970)',
'Silence of the Lambs, The (1991)',
'Manchurian Candidate, The (1962)',
'Bridge on the River Kwai, The (1957)']
{% endhighlight %} 

And now lets recommend top 20 movies to a female over 56 unknown user:
{% highlight python %}
woman_embeddings = model.embeddings(torch.tensor(9753,device=device))
over56_embeddings = model.embeddings(torch.tensor(9752,device=device))
metadata_embeddings = woman_embeddings+over56_embeddings
dot_results = movie_bias.squeeze()+(metadata_embeddings*movie_embeddings).sum(1)
[index_to_title[movies_tensor[i.item()].item()]
 for i in dot_results.argsort(descending=True)][:20]
{% endhighlight %} 

{% highlight python %}
['Gone with the Wind (1939)',
'To Kill a Mockingbird (1962)',
"Schindler's List (1993)",
'Some Like It Hot (1959)',
'African Queen, The (1951)',
'Color Purple, The (1985)',
'Grapes of Wrath, The (1940)',
'Room with a View, A (1986)',
'Casablanca (1942)',
'Wizard of Oz, The (1939)',
"Sophie's Choice (1982)",
'Wrong Trousers, The (1993)',
'West Side Story (1961)',
'Amadeus (1984)',
'Cinema Paradiso (1988)',
'Strangers on a Train (1951)',
"Singin' in the Rain (1952)",
'Shall We Dance? (Shall We Dansu?) (1996)',
'Rain Man (1988)',
'Fargo (1996)']
{% endhighlight %} 

Seems about right ¯\\_(ツ)_/¯  

Serve Recommender System with Elasticsearch
----------------
