# MixSIS: Mixed-supervised instance segmentation

[Christian Wilms](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/wilms.html), [Emre Gülsoylu](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/guelsoylu.html), [Sven Schreiber](https://svenschreiber.github.io/), and [Simone Frintrop](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/frintrop.html)

This repository contains data, models, and code for our BMVC'26 paper **MixSIS: Mixed-supervised instance segmentation**, including the dataset splits for our newly introduced MixSIS task. 

[[Paper](https://www.inf.uni-hamburg.de/en/inst/ab/cv/media/wilms-etal-bmvc2026.pdf)], [[Supplementary Material](https://www.inf.uni-hamburg.de/en/inst/ab/cv/media/wilms-etal-bmvc2026-supp.pdf)], [[Video](https://www.youtube.com/watch?v=iN0cQe5eE0Q)], [[Poster](https://www.inf.uni-hamburg.de/en/inst/ab/cv/media/wilms-etal-bmvc2026-poster.pdf)]

In traditional fully-supervised instance segmentation (FSIS) and weakly-supervised instance segmentation (WSIS), training usually uses a single annotation level, such as masks, boxes, points, or image-level annotations across the entire dataset. Since annotations have [different costs](https://arxiv.org/abs/2303.15062) (manual annotation effort), it is preferable to minimize the effort and optimize the annotation scheme. For instance, not every instance needs costly mask or box supervision. The person in Figure 1, which is well captured from a single point, is an example of this. In contrast, the bench in Figure 1 needs at least box supervision for a proper annotation. The reason here is that the centroids of both objects are within the person.

<img width="1696" height="629" alt="git_fig_example" src="https://github.com/user-attachments/assets/9d2ac60c-6eb1-4224-8176-5dab71b6c555" />

Taking advantage of that, we propose a new task: **MixSIS: Mixed-supervised instance segmentation**. In MixSIS, annotation levels are mixed across the dataset and even within an image. Here, we use the annotation levels masks, boxes, points, and image-level annotations. This flexible mix allows us to optimize the annotation process and, as a consequence, reduce annotation time. Figure 2 highlights the MixSIS idea (right), compared to traditional FSIS and WSIS task formulations  leveraging only a single annotation level (left).

<img width="1699" height="793" alt="git_fig_tasks" src="https://github.com/user-attachments/assets/fa725115-77f2-43cc-8137-8f9245926ee6" />

We address MixSIS with our baseline system, AnnsMixer, combining well-known [CAM](https://arxiv.org/abs/1512.04150), [Mask R-CNN](https://arxiv.org/abs/1703.06870), and [SAM](https://arxiv.org/abs/2304.02643). To evaluate AnnsMixer on the MixSIS task, we define three random mixes across the [COCO dataset](https://cocodataset.org/#home). In each mix, the instances in the COCO training set are randomly assigned to the four annotation levels. The size of the splits per level changes between the mixes. Table 1 shows the results of AnnsMixer and several WSIS baselines for the three proposed mixes. AnnsMixer clearly outperforms all baselines despite a simple architecture. Note that the paper also offers results of AnnsMixer on the traditional WSIS tasks, and three case studies highlighting the practical relevance of MixSIS reducing annotation time or improving detection results.

<img width="1373" height="664" alt="git_tab_mixSIS" src="https://github.com/user-attachments/assets/813cdaa9-ac0c-4dfe-914a-e8febbb279b9" />

## MixSIS Task
### Dataset Splits for COCO

We offer a `json` file per mix introduced in the paper to evaluate MixSIS. Each mix contains the annotation IDs of COCO's train2017 split assigned to the annotation levels. Technically, each `json` contains a `dict` with the keys `'mask'`, `'box'`, `'pts'`, and `'img'` mapping to a list of annotation IDs.

| Mix Name | Dataset Split |
|----------|---------------|
| Mix 1 | [Link to `json`](https://cloud.uni-hamburg.de/s/8dsji768gKEBeBE) |
| Mix 2 | [Link to `json`](https://cloud.uni-hamburg.de/s/N7sZWtd5x4pLJmE) |
| Mix 3 | [Link to `json`](https://cloud.uni-hamburg.de/s/oRpadZyXENqLxns) |

## AnnsMixer
### System

AnnsMixer is mainly built from off-the-shelf components, namely Mask R-CNN and SAM. To apply SAM given the geometric prompts, we use a variation of the script `applySAM.py`, provided by [the SOS git](https://github.com/chwilms/SOS_segment-anything). Given the SAM segments, we filter and merge the segments with the original mask annotations using the `combineAnnotations_cli.py` script, provided in this git repository. With the resulting annotations, we train Mask R-CNN with default settings.

### Model Zoo
In the following, we provide pseudo mask annotations generated in AnnsMixer, trained AnnsMixer models (Mask R-CNN part only), and AnnsMixer results on MixSIS and WSIS tasks.


#### MixSIS

| Mix Name | AnnsMixer Results [mAP] | Resources |
|----------|-----------------------|-----------------|
| Mix 1 | 0.323 | [Annotations](https://cloud.uni-hamburg.de/s/GCfbf8ip6XQqW2z), [trained model](https://cloud.uni-hamburg.de/s/W9sS2X9bxGx5p4f), [results](https://cloud.uni-hamburg.de/s/8QagBTGCAGE9BiS) |
| Mix 2 | 0.295 | [Annotations](https://cloud.uni-hamburg.de/s/2AzXPY2dNn7Lo3q), [trained model](https://cloud.uni-hamburg.de/s/5rs4NNcWKPq8Zcj), [results](https://cloud.uni-hamburg.de/s/Re7afk9eHwmtRXL) |
| Mix 3 | 0.266 | [Annotations](https://cloud.uni-hamburg.de/s/9rCBLAe5dCkWDog), [trained model](https://cloud.uni-hamburg.de/s/yoBioyzDaMepcgt), [results](https://cloud.uni-hamburg.de/s/GTprtdKacHNLaYz) |

#### WSIS

| Annotation Type | AnnsMixer Results [mAP] | Resources |
|----------|-----------------------|-----------------|
| Box | 0.345 | [Annotations](https://cloud.uni-hamburg.de/s/EjQJYjdKnD2ieQj), [trained model](https://cloud.uni-hamburg.de/s/9HfeWjtfyAEoz4g), [results](https://cloud.uni-hamburg.de/s/kKbK29zqcwXLXYY) |
| Point | 0.262 | [Annotations](https://cloud.uni-hamburg.de/s/JbkcTMTd4DXLEfY), [trained model](https://cloud.uni-hamburg.de/s/wnEZdfKzRWbP89e), [results](https://cloud.uni-hamburg.de/s/4YqFkZ7NWNpoRna) |
| Image | 0.220 | [Annotations](https://cloud.uni-hamburg.de/s/mwgPzCkZLTWNWBb), [trained model](https://cloud.uni-hamburg.de/s/apC45236F5RP25c), [results](https://cloud.uni-hamburg.de/s/4DWRXNQrJ23qWoj) |
