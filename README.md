# Feature Alignment: Rethinking Efficient Active Learning via Proxy in the Context of Pre-trained Models

Official code implementation for efficient active learning method, ASVP.
This code supports various sample selection and model training combinations, such as (1) Standard active learning via fine-tuning (FT) / linear-probing then fine-tuning (LP-FT): fine-tuning the pre-trained model at each active learning iteration; (2) SVPp (Selection via proxy based on pre-trained features): selecting samples via proxy model (MLP classifier with pre-trained features inputs) during all active learning iterations, after that fine-tuning model and evaluating it; (3) ASVP (aligned selection via proxy): initially, the pre-trained features are used as inputs for proxy model, an indictor based on LogME and/or PED is used to detect if the pre-computed feature is needed to update. If yes, fine-tuning the pre-trained model and updating pre-computed features. After active learning finishes, fine-tuning the pre-trained model to evaluate the final performance. 

What can we expect?
(1) Standard active learning: Good AL performance, i.e. label efficiency, while long AL sampling time especially for large-scale model.
(2) SVP: moderate AL performance, while fast AL sampling
(3) ASVP: keep AL performance with (1) and increasing marginal AL sampling time compared with (2).

# Installation 
 coming soon
 
 environment：

 data:

 pre-trained models:


# Scripts
We provide 3 example scripts in scripts folder for (1) standard AL (FT), (2) SVPp, (3) ASVP on ImageNet, other scripts will be released soon.
