---
layout: post
title:  "Movie Recommender from Spark 3.0.0 to Elasticsearch"
excerpt: "Train and serve a movie recommender using spark 3.0.0 mllib, factorization machines and elasticsearch "
date:   2020-07-04 00:00:00 +0200
categories: [recsys, spark, elasticsearch]
hide: true
---

Spark 3.0.0 release adds a new [Factorization Machines](https://issues.apache.org/jira/browse/SPARK-29224) regression model to its mllib and this is a greate opportunity (for me) to revisit my [first]({% post_url 2020-02-18-fm-torch-to-recsys %}) blog on factorization machines and this time train the model using spark.  
In this post I will prepare the movielens dataset as input to sparks new FM model, train the FMRegressor and feed the documents to elasticsearch using the elasticsearch-spark library.  
If you are not familiar with factorization machines you should start [here]({% post_url 2020-02-18-fm-torch-to-recsys %}).  
The full code in my [github](https://github.com/yonigottesman/spark_fm).


Prepare Data
====
The input to a Factorization Machines model is a list of events where each row represents a user rating a movie together with all the other features - age, gender and occupation. For example:

![movielens_input]({{ "/assets/movielens_input.jpg" | absolute_url }})


The FMRegressor I'm using from spark MLlib expects this list to be represented by a dataframe with a rating (label) column and a features column that is a sparse vector. For example

|rating|features                                            |
|------|----------------------------------------------------|
|4     |(9953,[0,6124,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|
|4     |(9953,[0,6129,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|
|4     |(9953,[0,6230,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|
|5     |(9953,[0,6238,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|

The sparse vector format is (\<vector_size\>,\<indices\>,\<values\>). All the vectors in our case have the same size 9953 which is the number of feature values and all the values in this case are 1. The first line is the event that user 0 rated movie 6124 with 4, the other features are 
* age 9929 (THE AGE)
* gender 9931 ( THE GENDER) 
* occupation 9943 (THE OCCUPATION)  

The following stages show how to transform the [movielens](http://files.grouplens.org/datasets/movielens/ml-1m.zip) dataset into this input dataframe.

First read the movies file and create a unique index for each title

```scala
val movies = spark.read.option("delimiter","::")
  .csv("ml-1m/movies.dat")
  .toDF("movie_id","title","genres")
    
val titleIndexer = new StringIndexer()
  .setInputCol("title")
  .setOutputCol("title_index")
  .fit(movies)
```

For the users I create a unique index for each user feature: user_id, gender, age, occupation. Each feature is in a different column so I can use a pipeline of indexers

```scala
val users = spark.read.option("delimiter","::")
  .csv("ml-1m/users.dat")
  .toDF("user_id","gender","age","occupation","zipcode")

val userFeaturesIndexers = Array("user_id","gender","age","occupation")
  .map(col => new StringIndexer().setInputCol(col).setOutputCol(col+"_index"))
val pipe = new Pipeline().setStages(userFeaturesIndexers).fit(users)
val usersIndexed = pipe.transform(users)
```

Join everything with ratings data

```scala
val ratings = spark.read.option("delimiter","::")
  .csv("ml-1m/ratings.dat")
  .toDF("user_id","movie_id","rating","time")
  .withColumn("rating",$"rating".cast(IntegerType))

val ratingsJoined = ratings
  .join(usersIndexed,Seq("user_id"))
  .join(moviesIndexed,Seq("movie_id"))

ratingsJoined.show(4)
```

|movie_id|rating|     time|gender|age|occupation|zipcode|user_id_index|gender_index|age_index|occupation_index|               title|              genres|title_index|
|-------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------
|    1193|      1|     5|978300760|     F|  1|        10|  48067|          0.0|         1.0|      6.0|            11.0|One Flew Over the...|               Drama|     2571.0|
|     661|      1|     3|978302109|     F|  1|        10|  48067|          0.0|         1.0|      6.0|            11.0|James and the Gia...|Animation|Childre...|     1818.0|
|     914|      1|     3|978301968|     F|  1|        10|  48067|          0.0|         1.0|      6.0|            11.0| My Fair Lady (1964)|     Musical|Romance|     2399.0|
|    3408|      1|     4|978300275|     F|  1|        10|  48067|          0.0|         1.0|      6.0|            11.0|Erin Brockovich (...|               Drama|     1096.0|


All our feature *_index columns are indices from 0 to size of feature. For example gender_index has values [0,1], title_index has values [0-3096] (3096 unique titles) but I need to shift each feature to start from the correct offset. First calculate the size of each feature using countDistinct

```scala
// movies feature sizes
val numMovies = moviesIndexed
  .select(countDistinct($"title_index").alias("numMovies"))
  .take(1)(0).getAs[Long]("numMovies")

// users feature sizes
val userfeatureColumns = Seq("user_id_index","age_index","gender_index","occupation_index")
val userFeaturSizesDf = usersIndexed
  .select(userfeatureColumns.map(c => countDistinct(col(c)).alias(c)): _*)
val userFeatureSizes = Map(userfeatureColumns
  .zip(userFeaturSizesDf.take(1)(0).toSeq.map(_.asInstanceOf[Long])):_*)
```


cumulative sum
