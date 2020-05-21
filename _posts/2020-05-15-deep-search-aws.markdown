---
layout: post
title:  "Deepfood - Image Retrieval System in Production"
excerpt: "Build and deploy a food image retrieval system with pytorch, annoy, starlette and AWS Elastic Beanstalk"
date:   2020-05-15 06:48:38 +0200
<!-- categories: deep-learning transfer-learning annoy-ann aws -->
---

In this post I will build and **deploy** a food image retrieval system. I will use pytorch to train a model that extracts image features, Annoy for finding nearest neighbor images for a given query image, starlette for building a web application and AWS Elastic Beanstalk for deploying the app. Lets begin!  
The full code is here: [github](https://github.com/yonigottesman/deepfood).  

System Architecture 
===================
![arch]({{ "/assets/image_search_arch.svg" | absolute_url }}){:height="100%" width="100%"}

workflow
=======
1. Train Models  
   * Train embedding extractor based on resnet34
   * Build Annoy nearest neighbor index
   * save to s3
2. Serve web app
   * Create docker image
   * Download annoy index + resnet model from s3 to docker image
   * Serve on Elastic Beanstalk
3. Accept client requests
   * Client sends an image
   * Web app transforms image to an embedding using pytorch model
   * Web app finds approximate nearest neighbors of embedding using Annoy
   * Return to client list of nearest images ids
   * Client downloads images from s3


Image Similarity
====
The idea behind an image retrieval system is having each image represented as an N dimensional vector (embedding).
And just like in word2vec, similar images will be close to one another in this N dimensional space.  
We need some kind of black box that takes an image and transforms it to an embedding, use this back box and transform our database of images to embeddings, then for every query image just find the closest embeddings from our database and return the images.  

Turns out deep neural networks are great black boxes for extracting embeddings! Each layer in a trained neural net learns to extract different features of an image, lower layers learn features such as "image contains a circle" and deeper layers learn features such as "image contains a dog" [[1](https://arxiv.org/abs/1311.2901)]. To use a trained neural net as a black box I will use pytorch hooks to extract the output of one of the last layers.  
The distance between two images is computed by: 
![similarity]({{- "/assets/similarity.svg" | absolute_url -}}){:height="100%" width="100%"}
I use a pre-trained resnet34 model and fine tune it to my specific domain - food images :hamburger:.
The reason I fine tune and not just use the pre-trained weights is because it was trained on imagenet with 1000 classes. The features extracted from deep layers could be "is there an eye", "is there a big grey thing", "are there ears" (probably :elephant:), but I need features more specific to food for example "is there a red round thing". Of course we don't know what features the model learns but later I will try to guess.  


Part I - Train Model
------
The full training code is in [this notebook](https://github.com/yonigottesman/deepfood/blob/master/notebooks/train_model.ipynb). Here I'm only going to show the important parts and skip the training loops, dataset creation etc..  
The dataset I use for fine tuning the model is [iFood](https://www.kaggle.com/c/ifood-2019-fgvc6/data), which contain food images like
![ifood]({{- "/assets/food.png" | absolute_url -}}){:height="100%" width="100%"}
First step is to download the resenet34 pretrained model and replace the last layer with a new one with 251 neurons (outputs) which is the number of classes in the ifood dataset.


```python
model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features # output of previous layer
model.fc = nn.Linear(num_ftrs, 251)
```

After about 20 epochs the top3 accuracy is around 0.84, this brings me to the middle of the scoreboard and is good enough for extracting embeddings. State of the art image similarity systems use a triplet loss function and dont train on a classification problem (like me) [2](https://arxiv.org/abs/1404.4661), [3](https://arxiv.org/pdf/1503.03832.pdf). 





Exctract Image Embeddings
======
To extract the embeddings of an image I need to read the values from the last layer of the nn. I use pytorch hooks mechanism to save the last layer (right after avgpool layer). 

![resnet]({{ "/assets/resnet_emb.png" | absolute_url }})

My EmbeddingExtractor registers a hook to the model on \__init\__. When get_embeddings() is called the image is passed through the network and the embedding will be waiting in self.embeddings
 field.

```python
class EmbeddingExtractor:
    def sniff_output(self,model, input, output):
        self.embeddings=output  
    def __init__(self,model):
        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 251)
        self.model = model.to(device)
        self.model.load_state_dict(torch.load('model.pt'))
        self.model.eval()
        layer = self.model._modules.get('avgpool')
        self.handle = layer.register_forward_hook(self.sniff_output)
    def get_embeddings(self, input):
        with torch.no_grad():
            self.model(input.to(device))
        return self.embeddings.squeeze(-1).squeeze(-1)
extractor = EmbeddingExtractor(model)
```
And thats our black box :white_square_button:.

What are Embeddings
===================
The neural network extracts for each image a 512 dimension vector. Each index represents a feature of the image. These features were learned automatically but we can still try to guess what each feature represents by feeding lots of images through the nework and displaying the images that maximize a specific feature (neuron).  
First I create a tensor of all the embeddings in my dataset (full code is [here]())

```python
all_emb=torch.tensor([])
for i,batch in enumerate(tqdm(dataloader)):
    emb = extractor.get_embeddings(batch['image'].to(device)).detach().cpu()
    all_emb=torch.cat([all_emb,emb],dim=0)
```
Then I choose feature index 10 (out of 512) and display top 10 images with highest value

```python
def display_best_images(feature_index):
    top_ten = sorted(range(len(all_emb)), key=lambda k: all_emb[k][feature_index].item(),reverse=True)[:10]
    top_images = torch.stack([train_dataset[i]['image'] for i in top_ten])
    imshow(vutils.make_grid(top_images, nrow=5, padding=2),f'index={feature_index}')
display_best_images(10)
```
![feature_10]({{ "/assets/feature_10.png" | absolute_url }})
Cool it looks like feature 10 is "image contains lots of stright thin lines". Remember the images dont have to look alike just share this single feature. You should run the code yourself and explore other features the network has learned.

Nearest Neighbors Search with Annoy
======================
After computing the embeddings for all images I use [Annoy](https://github.com/spotify/annoy) to build an index of all embeddings. Annoy is used to search embeddings similar (euclidean distance) to the query embedding in the index. It returns the approximate nearest neighbor but runs faster than comparing to all embeddings in the index. 

Part II - Deploy Application
------

Create Index & Upload to S3
======
I create the index from two datasets: [iFood](https://www.kaggle.com/c/ifood-2019-fgvc6/data) wich is the data I trained on, and [Food 101](https://www.kaggle.com/dansbecker/food-101). The id of each image is the filename in this format  \<id\>.jpg. I use [this]() script to downlaod the datasets, put all the images in the same folder and change all the filenames to ids. After I have a folder with all the images i can create a pytorch dataset and iterate through all the images, extracting an embedding and inserting to the annoy index:
```python
t = AnnoyIndex(512, 'euclidean')
for batch in tqdm(dataloader):
    embeddings = extractor.get_embeddings(batch[0])
    for i in range(len(batch[2])):
        emb = embeddings[i]
        img_id = os.path.basename(batch[2][i]).split('.')[0]
        t.add_item(int(img_id),emb)
t.build(5) # 5 trees
t.save('tree_5.ann')
```
Im building the annoy index with 5 trees, this is a configuration that tradeoffs size of index, speed and accuracy.
After index is saved I can upload the index and the resnet34 model to s3 later to be read by the web app:
```bash
aws s3 cp --acl public-read model.pt s3://deepfood/
aws s3 cp --acl public-read tree_5.ann s3://deepfood/
aws s3 cp --acl public-read index_images  s3://deepfood/ --recursive
```
You will have to install the aws cli and set permissions to write to s3 (More on this later).

Web App
=======
This is the application that binds everything together, gets user queries, searches nearest neighbors and returns the result. Im using [starlette](https://www.starlette.io/) which is a microframwork based on python asyncio but in this case there is no reason not to use flask which is more popular.
<!-- . asyncio frameworks are better if im planning to do async requests to other services (redis, sql...) but in this case its all cpu bound. The only reason I chose starlette is because I wanted to learn how to use it. You can read about Armin Ronacher thoughts on this [here](https://lucumr.pocoo.org/2020/1/1/async-pressure/) -->

The full code can be found [here](https://github.com/yonigottesman/deepfood/tree/master/deepfood_service) and I will go only over the important parts.  

routes.py contains the endpoints to the application, the important route is '/search' which accepts a POST request containing the image:

```python
@app.route('/search',methods=['POST'])
async def search(request):
    data = await request.form()
    content = data['image'].split(';')[1]
    image_encoded = content.split(',')[1]
    image_bytes = base64.decodebytes(image_encoded.encode('utf-8'))
    image = Image.open(io.BytesIO(image_bytes))
    embedding = extractor.get_embeddings(image)
    result_ids = ann_index.get_nns_by_vector(embedding, 9)
    urls = [f'https://deepfood.s3-us-west-2.amazonaws.com/ifood/{i}.jpg' for i in result_ids]    
    result = {'urls':urls}
    return JSONResponse(result)

```
The code extracs the image from the request, computes embeddindgs using our trained resnet, calls annoy search function and returns a list of urls for the frontend to display.


index.html is the front-end html, nothing special here except a little trick to get faster results. Instead of sending the full image (1-3MB) I added a script that resizes to our model input size 224*224 so that the image is sent mutch faster.



Docker
===
Easyest and most standard way of deploying an application is packaging it in a docker container. To do it I need a Dockerfile that is a set of inscructions to build the image.

```shell
FROM tiangolo/uvicorn-gunicorn:python3.7
WORKDIR /app
COPY requirements.txt /app
EXPOSE 80
COPY ./app /app
RUN mkdir app/models/
RUN wget --output-document=app/models/model.pt  https://deepfood.s3-us-west-2.amazonaws.com/model.pt
RUN wget --output-document=app/models/index.ann https://deepfood.s3-us-west-2.amazonaws.com/tree_5.ann
RUN pip install -r requirements.txt 
```
The only interesting operations here are 'RUN wget' that download the models we saved to s3 into the image so that the application can use them once its running. The others lines just copy everything, install requirements and run the application (running code here tiangolo/uvicorn-gunicorn).

to build the image and run it localy:
```shell
docker build -t myimage ./
docker run -d --name mycontainer -p 80:80 myimage
```
application is listening on port 80. and can be accessed in localhost:80.

Deploy - AWS Elastic Beanstalk
=====
Deploying on Beanstalk is easy peasy once you have a dockerfile. The only thing you need is run the eb-cli tool to init a new application:
```shell
eb init -p docker deepfood
```
Add some configurations to .ebextensions/custom.config so that the instance we get has enough memory and starage space to use our models:
```shell
aws:autoscaling:launchconfiguration:
  InstanceType: t2.large
  RootVolumeType: standard
  RootVolumeSize: "16"

```
And create a new environment with the application
```shell
eb create deepfood-env
```
More info [here](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/single-container-docker.html)

Thats it! just run 'eb open' to open a browser with link to your online application!. 



<script src="https://utteranc.es/client.js"
        repo="yonigottesman/yonigottesman.github.io"
        issue-term="pathname"
        label="comment"
        theme="github-light"
        crossorigin="anonymous"
        async>
</script>