---
layout: post
title:  "*DRAFT* Controllable Generation of Fixed GANs"
excerpt: "Learn how to edit the latent space to control semantic features of gan output"
date:   2020-11-17 00:00:00 +0200
categories: [pytorch,gan,wgan]
permalink: /2020/11/17/wgan-controllable-generation.html/
hide: true
---

<!-- Mathjax Support -->
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>



Conditional GAN architectures such as cgan train with class information together with the real images, a trained generator is then feeded with a random noise vector together with a class to generate an image of that class. But what if you have a fixed gan trained *without* additional class information, is it still possible to make the generator generate images of a specific class? according to this paper: [Interpreting the Latent Space of GANs for Semantic Face Editing](https://arxiv.org/abs/1907.10786), YES you can!  
In this post ill show how to turn a fixed unconstrained GAN to a controllable GAN by moving around the latent space in the right direction.


{% include note.html 
    content="I learned these ideas in the deeplearning.ai GANs [specialization](https://www.coursera.org/specializations/generative-adversarial-networks-gans?utm_source=deeplearningai&utm_medium=institutions&utm_campaign=DLWebGANsMain)." %}

Latent Space Interpolation
==
The generator of a trained GAN is a function $$ g:Z \to X $$ where $$ Z \subseteq  \mathbb{R}^d $$ is the $$d$$-dimensional latent space from which a noise vector is drawn and $$X$$ is the image space. Whats cool about this mapping is that small changes in a random vector $$z$$ correspond to small changes in the generated image. To visualize this coolness I can draw two vectors $$z1$$ and $$z2$$ from the latent space and display the generators output of vectors on the linear interpolation between them:

```python
z1 = torch.randn(1, 100, 1, 1, device=device)
z2 = torch.randn(1, 100, 1, 1, device=device)

fakes = []
for i in torch.linspace(0,1,10):
    with torch.no_grad():
        fake = generator(z1*i+z2*(1-i))
    fakes.append(fake)

mymshow(utils.make_grid(torch.cat(fakes,0),nrow=10,normalize=True),fig_size=[100,100])
```

![interpolation]({{ "/assets/controll_gan/interpolation.png" | absolute_url }})

$$z2$$ is the leftmost image and $$z1$$ is the rightmost image and you can see a smooth transition between them. 


Controllable Generation
== 
Just like with image interpolation, controllable generation also takes advantage of the fact that small changes in the latent space correspond to small changes in the generator output, except instead of moving on an interpolation between two vectors we move in a direction that only changes a single feature of the image. 
For example, if the output of the generator for a vector $$z1$$ is a man without glasses, and we want to generate a man with glasses, we can move in a direction $$n$$ that (nearly) doesn't change anything in the image except adding the man glasses: $$ z1\_new=z1+\alpha n$$ ($$\alpha$$=step size).  
The question is how do we even know there is a direction that will only add the man glasses without changing anything else? If this directions exists how do we find it?

Latent Space Separation
==
The paper makes the assumption that for any binary feature (*e.g.*, glasses\no-glasses, smiling\not-smiling, male\female) there  exists  a  hyperplane  in  the latent space serving as the separation boundary. For example if $$Z \subseteq \mathbb{R}^2$$ then there exists a line that separates points that will be generated with or without a smile:

![separation]({{ "/assets/controll_gan/separation.png" | absolute_url }})

Starting with a vector $$z$$ and moving in the latent space, the feature we are trying to change will remain the same as long as we are on the same side of the hyperplane. When the boundary is crossed, the feature turns into the opposite.

Classifier Gradients
==
