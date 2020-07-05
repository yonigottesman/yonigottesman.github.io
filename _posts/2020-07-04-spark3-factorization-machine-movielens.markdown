---
layout: post
title:  "*DRAFT* Movie Recommender from Spark 3.0.0 to Elasticsearch"
excerpt: "Train and serve a movie recommender using spark 3.0.0 mllib, factorization machines and elasticsearch "
date:   2020-07-04 00:00:00 +0200
categories: [recsys, spark, elasticsearch]
hide: true
permalink: /2020/07/04/spark3-fm-movielens.html/
---

Spark 3.0.0 release adds a new [Factorization Machines](https://issues.apache.org/jira/browse/SPARK-29224) regression model to its mllib and this is a greate opportunity (for me) to revisit my [first]({% post_url 2020-02-18-fm-torch-to-recsys %}) blog on factorization machines and this time train the model using spark.  
In this post I will prepare the movielens dataset as input to sparks new FM model, train the FMRegressor and feed the documents to elasticsearch using the elasticsearch-spark library.  
If you are not familiar with factorization machines you should start [here]({% post_url 2020-02-18-fm-torch-to-recsys %}).  
The full code in my [github](https://github.com/yonigottesman/spark_fm).

Prepare Data
====
The input to a Factorization Machines model is a list of events where each row represents a user rating a movie together with all the other features - age, gender and occupation. For example

![movielens_input]({{ "/assets/fm_spark_imput.png" | absolute_url }})
*figure 1*

Sparks mllib FMRegressor expects this list to be represented by a dataframe with a rating (label) column and a features column that is a sparse vector like this:

```
+----------------------------------------------------+------+
|features                                            |rating|
+----------------------------------------------------+------+
|(9953,[0,8611,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|5     |
|(9953,[0,7858,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|3     |
|(9953,[0,8439,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|3     |
|(9953,[0,7136,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|4     |
|(9953,[0,6617,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|5     |
+----------------------------------------------------+------+
```

The sparse vector format is (\<vector_size\>,\<indices\>,\<values\>). All the vectors in this case have the same size 9953 which is the number of feature values and all the values in this case are 1. The first line is the event that user 0 rated movie 8611 with 5, the other features are 
* age 9929 (THE AGE)
* gender 9931 - Male 
* occupation 9943 (THE OCCUPATION)  

The following stages show how to transform the [movielens](http://files.grouplens.org/datasets/movielens/ml-1m.zip) dataset into this input dataframe.

Read movies file and create a unique index for each title

```scala
val movies = spark.read.option("delimiter","::")
  .csv("ml-1m/movies.dat")
  .toDF("movie_id","title","genres")
    
val titleIndexer = new StringIndexer()
  .setInputCol("title")
  .setOutputCol("title_index")
  .fit(movies)
```

Read users file and create a unique index for each user feature: user_id, gender, age, occupation. Each feature is in a different column so I use a pipeline of indexers

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

ratingsJoined.select((featureColumns:+"rating").map(col):_*).show(4)
```
```
+-------------+-----------+---------+------------+----------------+------+
|user_id_index|title_index|age_index|gender_index|occupation_index|rating|
+-------------+-----------+---------+------------+----------------+------+
|          0.0|     2571.0|      6.0|         1.0|            11.0|     5|
|          0.0|     1818.0|      6.0|         1.0|            11.0|     3|
|          0.0|     2399.0|      6.0|         1.0|            11.0|     3|
|          0.0|     1096.0|      6.0|         1.0|            11.0|     4|
+-------------+-----------+---------+------------+----------------+------+
```
This is the list of events but each column contains indices ranging from 0 to size of feature - gender_index has values [0,1], title_index has values [0-3096] (3096 unique titles) etc. I need to shift the columns values so that each feature starts from the correct offset just like in figure 1. To add the offset, first calculate the size of each feature and then calculate the cumulative sum up until each feature:

Calculate the size of each feature using countDistinct

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
val featureSizes = userFeatureSizes + ("title_index"->numMovies)
print(featureSizes)
```
```
Map(user_id_index -> 6040, age_index -> 7, title_index -> 3883, occupation_index -> 21, gender_index -> 2)
```
Calculate cumulative sum until each feature using scanLeft

```scala 
val featureOffsets = Map(featureColumns
  .zip(featureColumns.scanLeft(0L)((agg,current)=>agg+featureSizes(current)).dropRight(1)):_*)
print(featureOffsets)
```
```
Map(user_id_index -> 0, age_index -> 9923, title_index -> 6040, occupation_index -> 9932, gender_index -> 9930)
```
Go over columns and add correct offset to column
```scala
val ratingsInput = ratingsJoined
  .select(featureColumns.map(name=>(col(name) + lit(featureOffsets(name))).alias(name)):+$"rating":_*)
ratingsInput.show(10)
```
```
+-------------+-----------+---------+------------+----------------+------+
|user_id_index|title_index|age_index|gender_index|occupation_index|rating|
+-------------+-----------+---------+------------+----------------+------+
|          0.0|     8611.0|   9929.0|      9931.0|          9943.0|     5|
|          0.0|     7858.0|   9929.0|      9931.0|          9943.0|     3|
|          0.0|     8439.0|   9929.0|      9931.0|          9943.0|     3|
|          0.0|     7136.0|   9929.0|      9931.0|          9943.0|     4|
+-------------+-----------+---------+------------+----------------+------+
```
Each column is now starting from the correct offset. Next convert each row to a sparse vector column and a label column
```scala
val createFeatureVectorUdf = udf((size:Int,
                                  user_id_index:Int,
                                  movie_index:Int,
                                  age_index:Int,
                                  gender_index:Int,
                                  occupation_index:Int) =>
Vectors.sparse(size,Array(user_id_index,movie_index,age_index,gender_index,occupation_index),Array.fill(5)(1)))
val data = ratingsInput
  .withColumn("features",createFeatureVectorUdf(lit(featureVectorSize)+:featureColumns.map(col):_*))
data.select("features","rating").show(10,false)
```
```
+----------------------------------------------------+------+
|features                                            |rating|
+----------------------------------------------------+------+
|(9953,[0,8611,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|5     |
|(9953,[0,7858,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|3     |
|(9953,[0,8439,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|3     |
|(9953,[0,7136,9929,9931,9943],[1.0,1.0,1.0,1.0,1.0])|4     |
+----------------------------------------------------+------+
```
Data is ready!

Train Model
====
split data to train and test
```scala
val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))
```
Create FMRegressor with embedding size 150 and fit on trainingdata
```scala
val fm = new FMRegressor()
      .setLabelCol("rating")
      .setFeaturesCol("features")
      .setFactorSize(150)
      .setStepSize(0.01)
      
val model = fm.fit(trainingData)
```
Evaluate rmse on testdata
```scala
val predictions = model.transform(testData)   
val evaluator = new RegressionEvaluator()
  .setLabelCol("rating")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
print(s"test rmse = $rmse")
```
```
test rmse = 0.88
```
[My pytorch model]({% post_url 2020-02-18-fm-torch-to-recsys %}) on the same data had 0.85 rmse so some more parameter tuning is probably needed here.  
Save model
```scala
model.write.overwrite().save("fmmodel")
```

