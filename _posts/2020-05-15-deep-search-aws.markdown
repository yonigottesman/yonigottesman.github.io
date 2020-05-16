---
layout: post
title:  "Deep Image Ranking Deployed on AWS"
excerpt: ""
date:   2020-05-15 06:48:38 +0200
<!-- categories: deep-learning transfer-learning annoy-ann aws -->
---

In this post I will build and deploy an image retrieval system. I will use pytorch to train a model that extracts image features, Annoy for finding nearest neighbor images for a given query image, starlette for building a web application and AWS Elastic Beanstalk for deploying the app. Lets begin!  
The full code is here: [github](https://github.com/yonigottesman/deepfood).  

System Architecture 
===================
[image]

Part I - Train Model
------
The idea behind an image retrieval system is having each image represented as an N dimensional vector (embedding).
And just like in word2vec, similar images will be close to one another in this N dimensional space.  
We need some kind of black box that takes an image and transforms it to an embedding, use this back box and transform our database of images to embeddings, then for every query image just find the closest embeddings from our database and return the images.  

Turns out deep neural networks are great black boxes for extracting embeddings! Each layer in a trained neural net learns to extract different features of an image, lower layers learn features such as "image contains a circle" and deeper layers learn features such as "image contains a dog" [[1](https://arxiv.org/abs/1311.2901)]. To use a trained neural net as a black box I will use pytorch hooks to extract the output of one of the last layers.  

I will use pytorch pre-trained 






What are Embeddings
===================

Nearest Neighbors Search with Annoy
======================


Part II - Deploy Application
------

Architecture image

Create Index & Upload to S3
======

Web App
=======
starlette why? why not flask - [link](https://lucumr.pocoo.org/2020/1/1/async-pressure/)


Deploy - AWS Elastic Beanstalk
=====

Docker
===




<script src="https://utteranc.es/client.js"
        repo="yonigottesman/yonigottesman.github.io"
        issue-term="pathname"
        label="comment"
        theme="github-light"
        crossorigin="anonymous"
        async>
</script>
