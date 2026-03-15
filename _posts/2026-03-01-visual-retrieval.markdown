---
layout: post
title: Scanned Pixels Beat OCR Text for Medical Document Retrieval
excerpt: I built a synthetic retrieval benchmark from MIMIC-III clinical notes to compare visual embedding models against the traditional OCR-then-embed pipeline.
date:   2026-03-01 00:00:00 +0000
categories: []
hide: false
---

Medical records often contain scanned text, whether they are physical documents scanned into the system or digital documents from which text cannot be extracted.
When building a search engine for medical documents that uses embeddings for retrieval (as part of a larger GenAI application), you have two options:

1. OCR documents to plain text and create an embedding from text
2. Use a new class of multimodal models to embed the raw scanned image

{% include note.html 
    content="Embeddings are only part of the story - medical retrieval should also use text search, and I'm not accounting for visual elements like charts." %}


[nemotron-colembed](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) was released last month and I wanted to see how well it handles raw scanned **medical** text compared to the OCR approach and to non-late-interaction models. Late interaction models store one embedding per token instead of a single embedding for the whole text, so you pay more on storage but gain more accuracy. These models are becoming popular especially for visual embedding tasks, as you can see in the [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) benchmark published in the [ColPali](https://arxiv.org/abs/2407.01449) paper.

This post describes my experiments evaluating these models in this domain, including the iterative process of creating my (tiny) dataset and LLM-judge. All the code to reproduce is available at [github.com/yonigottesman/visual-retrieval](https://github.com/yonigottesman/visual-retrieval).

## Problem Definition
The problem I'm solving is of a physician asking questions on a patient with thousands of documents, and the search engine should return the documents containing the answer. For example, a question could be "What is the patient's LVH measurement value?" and the relevant document should contain the answer.

## Synthetic Dataset

### Queries 
I don't have any real production traces of user queries, and there is no public dataset for what I'm looking for, so I need to create a synthetic dataset. The common approach is giving an LLM a document and asking it to generate questions. I'm using the [mimic-iii](https://physionet.org/content/mimiciii/1.4/) dataset for the notes and started with the [ViDoRe-V3](https://arxiv.org/abs/2601.08620) prompt for generating synthetic queries, iterating on patient documents and sending the doc + prompt to `gemini-3-flash` to generate physician queries.  
The vanilla [prompt](https://arxiv.org/html/2601.08620v1#A11.F20) worked poorly. The generated queries were garbage and didn't look like what I wanted. I needed a way to quickly review the generated queries, so I asked Claude Code [CC] to create a [function](https://github.com/yonigottesman/visual-retrieval/blob/master/generate_queries.py#L333) that takes all the generated queries and creates a single-page HTML so I can review and save only relevant queries. Here is the [generated HTML](https://storage.googleapis.com/visual-retrieval/review_queries.html):

<iframe id="review-iframe" data-src="https://storage.googleapis.com/visual-retrieval/review_queries.html" width="100%" height="600" style="border: 1px solid #ddd; border-radius: 4px;" title="Query review"></iframe>
<script>
(function() {
  var iframe = document.getElementById('review-iframe');
  var observer = new IntersectionObserver(function(entries) {
    if (entries[0].isIntersecting) {
      iframe.src = iframe.dataset.src;
      observer.disconnect();
    }
  }, {rootMargin: '200px'});
  observer.observe(iframe);
})();
</script>
<br>
  
At first I thought I would use this tool to manually pick only the best queries, but the initial ViDoRe prompt didn't produce any good ones. 
I needed to improve the query generation prompt! Instead of manually changing the prompt, running the full generation, and viewing the results, I gave CC this iteration task so it could iterate a few times before I reviewed the results. I gave CC this meta-prompt:

{% include prompt.html 
content="You are helping create synthetic queries for a medical document retrieval system\.\.\.
The questions should be from a perspective of a physician trying to gather information on the patient. optimize prompt slowly and rerun check results and repeat. once you feel its better show me again. either way dont do more than 10 iterations without showing me.
#### Here are some good and bad examples:
**GOOD**: What bilirubin level prompted the initiation of double phototherapy for the infant? — general physician perspective  
**BAD**: How soon after leaving the hospital will the follow-up home visit occur? — too administrative, a doctor wouldn't ask this on a full patient chart"
%}

In each iteration, CC tweaked the prompt used by Gemini to generate queries, ran the full query generation script, reviewed the resulting queries, and fixed the Gemini prompt again. I continued iterating with CC, gave it more good and bad examples, and let it iterate on its own. I ended up splitting the queries into three types:   

**EASY** - An easy query uses the EXACT SAME words, abbreviations, and terminology
       that appear in the document. Every key term in the query must appear
        verbatim in the document text. A simple keyword search (ctrl+F) on the
        document would match the important words in the query.

**MEDIUM** - A medium query asks the SAME KIND of simple, direct clinical question as
        an easy query, but makes small wording changes so that a simple ctrl+F
        keyword search would NOT match. Specifically:
        - Expand abbreviations to their full form (e.g. "bili" → "bilirubin",
          "resp" → "respiratory", "abx" → "antibiotics", "dopa" → "dopamine")
        - Or swap a word for a common synonym (e.g. "feeding" → "nutrition",
          "meds" → "medications", "labs" → "laboratory values")  

**HARD** - A hard query rephrases or uses medical synonyms so that a retrieval system
        needs SEMANTIC understanding, not just keyword matching, to find the answer.
        The wording in the query should differ from the wording in the document.

See the [generation prompt](https://github.com/yonigottesman/visual-retrieval/blob/32decb03f8a37e6ac8c6070c8569e417991ba23c/generate_queries.py#L77) and the HTML above to get a feel for how the types differ.
I can now run [generate_queries.py](https://github.com/yonigottesman/visual-retrieval/blob/master/generate_queries.py) which iterates all patient documents, sends them to Gemini with the generation prompt, and generates all the queries I want. I added a dedup step since these documents contain lots of redundancy.

### Scanned Docs
I'm using mimic-iii which contains text documents, so to transform them into scanned PDFs I add some noise and render the text as a "scanned" PDF.
[create_pdfs.py](https://github.com/yonigottesman/visual-retrieval/blob/master/create_pdfs.py) iterates all patient documents and creates a PDF version of each. Here is an example text and PDF:

<div style="display: flex; gap: 1rem; margin: 1rem 0;">
  <div style="flex: 1; min-width: 0;">
    <span style="display: block; margin-bottom: 0.25rem; font-size: 0.9em; color: #666;">Original text</span>
    <iframe src="https://storage.googleapis.com/visual-retrieval/note_1906543_original.txt" title="Original text" style="width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 4px;"></iframe>
  </div>
  <div style="flex: 1; min-width: 0;">
    <span style="display: block; margin-bottom: 0.25rem; font-size: 0.9em; color: #666;">Scanned PDF</span>
    <iframe src="https://storage.googleapis.com/visual-retrieval/note_1906543.pdf" title="Scanned PDF" style="width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 4px;"></iframe>
  </div>
</div>

### OCR 
To compare visual embedding on the scanned PDF with the straightforward OCR + text embedding approach, I first need to run OCR on the scanned PDFs I just created.
My CLAUDE.md has instructions for Claude to spawn a [vllm](https://vllm.ai/) instance with [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) running on it. Once the instance is running, I run [batch_ocr.py](https://github.com/yonigottesman/visual-retrieval/blob/428d52f840762d76402b97024e16b58d9ed17dc2/batch_ocr.py) which takes all the scanned PDFs, performs OCR on each, and writes the text result.

### Final Dataset
Now I have the dataset I'm going to work with. I have a [queries.json](https://github.com/yonigottesman/visual-retrieval/blob/master/cached_deduped.json) with a list of queries, answers, and the document each was taken from. For each document there are three versions:
1. original text 
2. scanned pdf (image)
3. OCR text

All documents are from a single patient, patient_id 16118 from the mimic-iii dataset. I chose this patient because he had many notes of different types. In total I have 1149 queries (441 easy, 434 medium, and 274 hard) across 727 documents. I removed documents that were too long (exceeding 1 page as a PDF) and documents too short (just a few words).

## Creating Embeddings
The next step is to create embeddings for all document types and queries with the different models I want to evaluate.
The models I am going to compare are:
* [nvidia/nemotron-colembed-vl-8b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) - Late interaction multimodal model. Embeddings for scanned images and OCR text.
* [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) - Multimodal model (no late interaction). Embeddings for scanned images and OCR text.
* [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) Text-only model. Embeddings for OCR text.
* [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) - Tiny (`20x` smaller than other models) text model. Embeddings for OCR text.
  
Ideally I would deploy a vllm instance on my gcloud for all these models and run everything against that. But unfortunately vllm does not support late-interaction models for image inputs yet. Also, for the other models I found some discrepancies between the original HuggingFace examples and running through vllm.  
All embeddings were instead computed with the standalone scripts in [generate_embeddings](https://github.com/yonigottesman/visual-retrieval/tree/master/generate_embeddings) running on a GPU instance.  
 
## Evaluation
The moment of truth! How good is each embedding model at retrieving relevant medical documents?
After generating all the embeddings for documents and queries, I created for each model a `similarity.npz` containing the similarity between each query and all documents. For regular embeddings I use [cosine similarity](https://github.com/yonigottesman/visual-retrieval/blob/32decb03f8a37e6ac8c6070c8569e417991ba23c/generate_embeddings/single_embedding_similarities.py#L47) and for colembed I compute [maxsim](https://github.com/yonigottesman/visual-retrieval/blob/32decb03f8a37e6ac8c6070c8569e417991ba23c/generate_embeddings/nemotron_similarities.py#L38) similarity.  

Now, given `similarity.npz` from the previous step, I want to compute standard retrieval metrics:
* hit - Fraction of queries with at least one relevant document in top-K
* precision - Average proportion of relevant documents in top-K
* ndcg - Normalized Discounted Cumulative Gain
* mrr - Mean Reciprocal Rank (1/rank of first relevant document)

The thing about these medical documents is that data is repeated again and again, so I cannot easily compute `recall` because I would need to know for each query the full list of relevant documents. That's why I'm sticking to metrics where all I need is the top-K list and whether each result is relevant or not.

### LLM as a Judge
Given a list of top-K results per query, I need to know which of the top-K documents actually answer the query. I can either manually annotate the results or give an LLM a (query, document) tuple and ask whether the document answers the query.
To get a good judgment prompt I used the same iterative technique with CC I used for the query generation prompt.
I had CC run Gemini with different queries and documents (some relevant, some not) and let it optimize the prompt by validating that the judge returned the correct True/False for each.
The resulting prompt is:

{% include prompt.html 
content="You are a medical information retrieval evaluator. Given a clinical query and retrieved documents, 
determine if each document contains information that directly answers or is highly relevant to the query.

A document is RELEVANT if:
- It contains specific information that answers the query
- It discusses the medical concept, condition, or treatment asked about in a way that addresses the query
- The answer to the query can be found or inferred from the document content

A document is NOT RELEVANT if:
- It is about the same patient but does not address the specific question
- It mentions related medical terms but does not contain the answer
- It is a completely unrelated document

Query: {query}

Documents:
{documents}

For each document, provide your judgment as JSON with a \"judgments\" array. 
Each item must have doc_id, relevant (boolean), and reason (brief explanation)."%}

### Results

<style>
  .tabs { display: flex; gap: 4px; margin-bottom: 1.5rem; }
  .tab { padding: 6px 16px; border-radius: 20px; border: 0.5px solid #ccc; font-size: 13px; cursor: pointer; background: transparent; color: #888; }
  .tab.active { background: #3266ad; color: #fff; border-color: #3266ad; font-weight: 500; }
  .metric-tabs { display: flex; gap: 4px; margin-bottom: 1rem; flex-wrap: wrap; }
  .mtab { padding: 4px 12px; border-radius: 12px; border: 0.5px solid #ddd; font-size: 12px; cursor: pointer; background: transparent; color: #888; }
  .mtab.active { background: #3266ad; color: #fff; border-color: #3266ad; }
</style>

<div style="padding: 1rem 0;">
  <div class="tabs">
    <button class="tab active" onclick="setDiff('easy')">Easy</button>
    <button class="tab" onclick="setDiff('medium')">Medium</button>
    <button class="tab" onclick="setDiff('hard')">Hard</button>
  </div>

  <div class="metric-tabs">
    <button class="mtab active" onclick="setMetric('hit')">Hit@10</button>
    <button class="mtab" onclick="setMetric('p10')">P@10</button>
    <button class="mtab" onclick="setMetric('ndcg')">NDCG@10</button>
    <button class="mtab" onclick="setMetric('mrr')">MRR</button>
  </div>

  <div style="position: relative; width: 100%; height: 280px;">
    <canvas id="mainChart"></canvas>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const data = {
  easy: [
    { label: 'Nemotron / scanned', hit: 100.0, p10: 0.7599, ndcg: 0.9628, mrr: 0.9860, color: '#3266ad' },
    { label: 'Nemotron / ocr',     hit: 100.0, p10: 0.7324, ndcg: 0.9478, mrr: 0.9665, color: '#5b8dd9' },
    { label: 'Qwen-VL / scanned',  hit: 98.9,  p10: 0.5751, ndcg: 0.8444, mrr: 0.8247, color: '#1d9e75' },
    { label: 'Qwen3 / ocr',        hit: 95.7,  p10: 0.5306, ndcg: 0.7956, mrr: 0.7679, color: '#85b7eb' },
    { label: 'Qwen-VL / ocr',      hit: 93.7,  p10: 0.4766, ndcg: 0.7671, mrr: 0.7392, color: '#5dcaa5' },
    { label: 'Gemma / ocr',        hit: 91.8,  p10: 0.4560, ndcg: 0.7297, mrr: 0.6904, color: '#888780' },
  ],
  medium: [
    { label: 'Nemotron / scanned', hit: 99.8, p10: 0.7385, ndcg: 0.9297, mrr: 0.9360, color: '#3266ad' },
    { label: 'Nemotron / ocr',     hit: 98.9, p10: 0.6859, ndcg: 0.9053, mrr: 0.9055, color: '#5b8dd9' },
    { label: 'Qwen-VL / scanned',  hit: 98.4, p10: 0.6037, ndcg: 0.8474, mrr: 0.8300, color: '#1d9e75' },
    { label: 'Qwen-VL / ocr',      hit: 95.9, p10: 0.5362, ndcg: 0.8093, mrr: 0.7924, color: '#5dcaa5' },
    { label: 'Qwen3 / ocr',        hit: 96.1, p10: 0.5680, ndcg: 0.8056, mrr: 0.7723, color: '#85b7eb' },
    { label: 'Gemma / ocr',        hit: 93.5, p10: 0.5055, ndcg: 0.7506, mrr: 0.7077, color: '#888780' },
  ],
  hard: [
    { label: 'Nemotron / scanned', hit: 97.8, p10: 0.6339, ndcg: 0.8670, mrr: 0.8684, color: '#3266ad' },
    { label: 'Nemotron / ocr',     hit: 96.4, p10: 0.5693, ndcg: 0.8185, mrr: 0.8109, color: '#5b8dd9' },
    { label: 'Qwen-VL / scanned',  hit: 95.3, p10: 0.5635, ndcg: 0.8019, mrr: 0.7755, color: '#1d9e75' },
    { label: 'Qwen-VL / ocr',      hit: 96.4, p10: 0.5248, ndcg: 0.7848, mrr: 0.7538, color: '#5dcaa5' },
    { label: 'Qwen3 / ocr',        hit: 95.6, p10: 0.5405, ndcg: 0.7766, mrr: 0.7398, color: '#85b7eb' },
    { label: 'Gemma / ocr',        hit: 92.0, p10: 0.4777, ndcg: 0.7242, mrr: 0.6780, color: '#888780' },
  ]
};

const metricKey = { hit: 'hit', p10: 'p10', ndcg: 'ndcg', mrr: 'mrr' };
const metricLabel = { hit: 'Hit@10 (%)', p10: 'P@10', ndcg: 'NDCG@10', mrr: 'MRR' };
const metricFmt = {
  hit: v => v.toFixed(1) + '%',
  p10: v => v.toFixed(4),
  ndcg: v => v.toFixed(4),
  mrr: v => v.toFixed(4),
};

let currentDiff = 'easy';
let currentMetric = 'hit';
let chart;

function getChartData() {
  const rows = data[currentDiff];
  const mk = metricKey[currentMetric];
  return {
    labels: rows.map(r => r.label),
    values: rows.map(r => r[mk]),
    colors: rows.map(r => r.color),
  };
}

function initChart() {
  const ctx = document.getElementById('mainChart').getContext('2d');
  const { labels, values, colors } = getChartData();

  const isHit = currentMetric === 'hit';
  const minVal = isHit ? 85 : 0.4;

  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => metricFmt[currentMetric](ctx.raw)
          }
        }
      },
      scales: {
        x: {
          min: minVal,
          grid: { color: 'rgba(128,128,128,0.1)' },
          ticks: {
            color: '#888',
            font: { size: 11 },
            callback: v => isHit ? v + '%' : v.toFixed(2)
          }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#888', font: { size: 12 } }
        }
      }
    }
  });
}

function updateChart() {
  const { labels, values, colors } = getChartData();
  const isHit = currentMetric === 'hit';
  chart.data.labels = labels;
  chart.data.datasets[0].data = values;
  chart.data.datasets[0].backgroundColor = colors;
  chart.options.scales.x.min = isHit ? 85 : 0.4;
  chart.options.scales.x.ticks.callback = v => isHit ? v + '%' : v.toFixed(2);
  chart.options.plugins.tooltip.callbacks.label = ctx => metricFmt[currentMetric](ctx.raw);
  chart.update();
}

function setDiff(d) {
  currentDiff = d;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  updateChart();
}

function setMetric(m) {
  currentMetric = m;
  document.querySelectorAll('.mtab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  updateChart();
}

initChart();
</script>

Nemotron on scanned images is clearly the best across difficulties and most metrics. It's surprising (to me) to see **text** retrieval perform best when the text is actually represented as image pixels. 
I created the [widget below](https://storage.googleapis.com/visual-retrieval/search.html?v=2) to show top-K results for all models and difficulties to easily debug errors.

<iframe src="https://storage.googleapis.com/visual-retrieval/search.html?v=2" width="100%" height="600" style="border: 1px solid #ddd; border-radius: 4px;" title="top-k"></iframe>    
<br>
 
Here are some interesting failing queries.  
**Nemotron scanned**   
"What medication was prepared to counteract the adverse effects of morphine?" - All top results contain "morphine". This is a hard question to capture in an embedding. The embedding somehow needs to represent both "morphine" and "counteract the adverse effects". This seems really hard to me.

**Nemotron OCR**  
"What was the infant's blood sugar reading after receiving insulin?" - Another hard embedding example. The embedding should capture "blood sugar" and "after receiving insulin". The top result does contain "insulin" but does not mention blood sugar. Interestingly, the scanned version actually did manage to capture this connection, which is really cool given this is not trivial. Looking at the document it's not straightforward to capture this connection, but Nemotron scanned does. Here is the top note for the scanned result:
```
...
GI/GU: Abdomen soft, flat, no loops. +BS. Unable to palpate testes in canal.
Genitilia appropriate for gestation. Voiding approximately 5.2cc/kg/hr.
UA as noted in labs.
Note trace glucose but have had increased glucose levels (see FEN).  No stools reported.
FEN: TF 220 now decreased to 200cc/kg/day.
IVF decreased from D10 to D5W.  Last glucose 11am 211.
Decreased from 290- insulin 0.05units given.  Starting TPN and IL today.  Lytes 149/3.6/115/23.  Recheck at 9pm.  UAC rate decreased from 1 to 0.8 to decrease Na amt infant receiving to 2.8 through this line.
...
```

## Final Thoughts
Nemotron is a really good model (according to this modest benchmark), but before jumping to use it there are some things to consider. Late interaction models take `10x` more space in your search engine because we store an embedding per token. Also, it's a big model; 8B is not trivial and must run on a GPU.  
I think the tiny `gemma` model has great results too. `Hit@10` is above 90% across all difficulties, which is great given I usually need just a single document that contains the answer. If I add BM25 text search and some query expansions done by an agent using retrieval as a tool, it might actually be very cost effective. A text model does require OCR, but I think OCR for plain text is essentially solved. Also, having a pipeline solution makes it easier to debug failures than a single end-to-end embedding model.



<script src="https://utteranc.es/client.js"
        repo="yonigottesman/yonigottesman.github.io"
        issue-term="pathname"
        label="comment"
        theme="github-light"
        crossorigin="anonymous"
        async>
</script>
