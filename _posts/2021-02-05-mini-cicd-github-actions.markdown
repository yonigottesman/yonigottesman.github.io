---
layout: post
title:  "*DRAFT* Mini CI/CD for Personal ML Projects"
excerpt: "Add github-actions to yout ML project to improve code quality and manage aws deployment."
date:   2021-02-05 00:00:00 +0200
categories: [cicd,mlops,github]
<!-- permalink: mini-cicd-github-actions.html/ -->
hide: true
---

If you are working in a big tech company, chances are you have a CI/CD pipeline built by engineers and devops for you. But what if you are working on a small side project, research or kaggle? You can still set up pretty decent CI/CD pipeline using [github-actions](https://github.com/features/actions). In this post I show how to take your ML project to the next level by:
* Automatically testing your code when a PR is created (pytest and pep8 compliant).
* Automatically running new models in PR on testset and printing a summary as a message.
* Serving model using FastAPI on aws and automatically deploying new models when PR is merged.

These steps should not be taken as all-or-none, for example even if your project has no ML you should still add automatic steps to test your code and pass it through linters. 

The example project with all these goodies is [here](https://github.com/yonigottesman/sentiment), and the model is served [here](http://sentiment-env.eba-pcptvp7e.us-west-2.elasticbeanstalk.com/predict?tweet=i%20have%20really%20good%20luck%20:%29)

Sentiment Project Structure
---
The example project I'm  working on is a tweet sentiment classifier. Given a raw tweet return if its sentiment is positive or negative.  
The project structure is:
```
├── sentiment
│   ├── __init__.py
│   ├── inference.py
│   └── processing.py
├── app
│   └── main.py
├── tests
│   ├── __init__.py
│   └── test_inference.py
├── Dockerfile
├── README.md
├── requirements.txt
├── train.py
├── eval.py
└── upload_model.py
```
**sentiment** - Core sentiment module. The inference code downloads the model binary from s3 when created.  
**app** - Serving module using FastAPI.  
**tests** - Some tests.  
**train.py** - Train the logistic regression classifier.  
**eval** - Run evaluation of model on testset.  
**upload_mode.py** - Upload model to s3.  


CI/CD Workflow
--

1. Model is happily served [here](http://sentiment-env.eba-pcptvp7e.us-west-2.elasticbeanstalk.com/predict?tweet=i%20have%20really%20good%20luck%20:%29).
2. Train and improve model.
3. Upload new model to s3.
4. Create pr with new code pointing to new model location.
5. Merge PR to main branch.
6. goto step 1.

Lets go over these steps one by one:


#### Train Model
[train.py](https://github.com/yonigottesman/sentiment/blob/main/train.py) training script creates a logistic regression classifier on tfidf features. The dataset is nltk.corpus twitter_samples containing 10K tweets. The code is self explanatory nothing more to say on this simple model :-).  
After I'm happy with the code changes and model performance I [upload](https://github.com/yonigottesman/sentiment/blob/main/upload_model.py) the model binary to s3 and create a PR with the new code. The new binary should not overwrite the old one, the name convention I am using here is model_v1.pkl so the version number is in the name. This is important because the code currently in production is coupled with the old version and wont work if it suddenly reads a new binary. The new code in the PR should point to the new model binary and after the PR is merged the new code+binary will be served.


#### Model Serving
[FastAPI](https://fastapi.tiangolo.com/) web framework running on aws elasticbeanstalk. Deploying is super easy using the [eb-cli](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/docker.html) and Dockerfile. From the root directory run:
```shell
eb init -p docker sentiment
eb create sentiment-env
eb open
```
Follow [this](https://aws.amazon.com/getting-started/hands-on/set-up-command-line-elastic-beanstalk/) tutorial to create access keys on your aws account so that the cli has permission to deploy.  
The FastAPI app expects the tweet to be in a query string, so sending "I have really good luck :)" looks like: 
[http://sentiment-env.eba-pcptvp7e.us-west-2.elasticbeanstalk.com/predict?tweet=i%20have%20really%20good%20luck%20:%29](http://sentiment-env.eba-pcptvp7e.us-west-2.elasticbeanstalk.com/predict?tweet=i%20have%20really%20good%20luck%20:%29)

#### Pull Request Checks
When a PR is created two github-action workflows are started:
1. [python-qulity.yml](https://github.com/yonigottesman/sentiment/blob/main/.github/workflows/python-quality.yml) - This workflow will run flake8 to enforce code style and run tests using pytest. For easier development you should use formatting tools such as yapf black and isort. These tools can be used directly in vscode and pycharm.  You can also add mypy and pylint to the workflow and enforce even better code quality.

2. [model-perfomance.yml](https://github.com/yonigottesman/sentiment/blob/main/.github/workflows/model-performance.yml) - This workflow will run the eval.py script, compare the result to main branch and comment a nice table in the PR [conversation](https://github.com/yonigottesman/sentiment/pull/15#issuecomment-775352911). The result looks like this:

![pr-comment]({{ "/assets/cicd/pr-comment.png" | pr-comment }})

You can edit the yml to display any table/result you like.

#### Merge PR
After the PR is approved and merged to main branch, [eb-deploy.yml](https://github.com/yonigottesman/sentiment/blob/main/.github/workflows/eb-deploy.yml) workflow kicks in and runs the eb-cli to deploy the new code. This workflow expects the repository to have two secrets which are the keys from before:
```
AWS_ACCESS_KEY_ID: $ { { secrets.AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY: $ { { secrets.AWS_SECRET_ACCESS_KEY }}
```
And thats it! new model and code are now being served.  

![happy]({{ "/assets/cicd/happy.jpg" | happy }})



Final Words
--
Adding these simple workflows can improve your productivity and code quality!

* Its easy to reproduce earlier results by checking out an earlier version that uses the matching binary.
* High code quality and testing is super important for your project to be readable by others.
* Deployment becomes something you set once and forget about it.
* In this post I did not talk about dataset versioning which is important to reproduce results. During a project life cycle the dataset changes, gets fixed and gets bigger. Each version of the code+binary should also be coupled with the version of the dataset so that checking out an early commit can reproduce the exact model with the same data.








<script src="https://utteranc.es/client.js"
        repo="yonigottesman/yonigottesman.github.io"
        issue-term="pathname"
        label="comment"
        theme="github-light"
        crossorigin="anonymous"
        async>
</script>
