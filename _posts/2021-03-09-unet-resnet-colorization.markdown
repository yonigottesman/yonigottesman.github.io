---
layout: post
title:  "Deep Learning Image Colorization"
excerpt: "Color black and white images using a Unet with Resnet backbone"
date:   2021-03-09 00:00:00 +0200
categories: [unet, vision]
<!-- permalink: mini-cicd-github-actions.html/ -->
hide: true
<!-- image: /assets/cicd/happy.jpg -->
---

[Unet](https://arxiv.org/abs/1505.04597) architectures have become one of the preferred methods for image segmentation, but just as convnets have evolved from basic Lenet to AlexNet->VGG->Inception->Resnet->DensNet, The Unet has also evolved to use these advanced techniques. One way to make "better" Unets is to take a known pretrained architecture and use it as the Unets encoder (backbone), connecting skip connections to the decoders.  
<!-- A nice thing about Unets is that they can be used to do any image-to-image translations tasks as shown in the [pix2pix](https://arxiv.org/abs/1611.07004) paper. Fastai has a nice [lesson](https://github.com/fastai/fastai/blob/2.2.7/dev_nbs/course/lesson7-superres.ipynb) on how to do super resolution (or any image translation) using its library.   -->

In this post I show how to build a Unet with a Resnet backbone using pytorch. This model will be trained to transform black and white images to RGB.  
All the code is my [git]()

Unet
--


Resnet BackBone
--

generic pytorch resnet architecture

![resnet]({{ "/assets/resunet/generic_resnet.png" | resnet }})<!-- {:height="50%" width="50%"} -->


![resunet]({{ "/assets/resunet/resunet.svg" | resunet }})<!-- {:height="50%" width="50%"} -->


Model
--

Loss
--


Data
--

Train
--

Results
--



