# Environment Instructions
To set up the Python environment and download the datasets, set up conda and run the following command:
```bash
./setup.sh
```

# Milestone 1
* Replicate the original paper
* Train various models with the dataset
    * Download the datasets
    * Use pre-implemented models (e.g. ResNet, SKLearn stuff)
    * Get baseline results
* Find interesting patterns of the data/model
    * PCA on the last layer of the encoding of the model (e.g. ResNet, autoencoder)
        * Plot the 2D embedding of the data
        * Plot the decision boundary of the model
* Potential aspects to improve with respect to the model or the training method

* What is the motivation?
    * We want to identify how well we can perform with lower quality data when human uncertainty is considered
    * We want to see if we can use the uncertainty to improve the performance of the model
    * Find a methodology that will improve training on low-quality data
        * On the way, find methods to automatically improve data quality (e.g. use PCA to clean up noise, use clustering to find similar images)
    * Data-centric AI -> can we use this as a way to improve training on low-quality data?

* For literature review:
    * Mention paper that we had before (the one we're replicating) -> Human Uncertainty makes Classification Robust

* TODO:
    * Model baselines
    * Find interesting patterns
    * Writing

# Milestone 2 (after the milestone submission)
* Identify some failure cases of the best baseline model
* Try out some potential solutions to improve the model performance
* The improvement can be in the aspect of model, dataset, training method, loss function, etc. 


# Final Goal
When I say taking "model architecture optimization, ensemble methods, or cross-dataset generalization" as the final goal of the project, I'm actually citing your last sentence of "Intended experiments". I think besides taking the CIFAR-10H dataset and training an existing model with that data, you can consider ways to improve the model architecture or try to ensemble different models/methods to increase the classification accuracy, or apply similar ideas to other datasets/applications. These are some ways to expand the scope of your project.


# Idea for Next Steps
* Train a model on CIFAR-10H to try predict human uncertainty
* Apply this model to a chunk of CIFAR-10 data to create a new dataset with predicted human uncertainty
* Train a model on the new dataset with predicted human uncertainty and compare the performance to the original model trained on CIFAR-10
    * Augment the hard labels into a distribution of soft labels
* Try to apply this idea to other datasets (e.g. CIFAR-100, ImageNet)

Previously: image / hard label -> classification (CIFAR10)
Baseline: (image -> soft label) -> (image / soft label -> classification) (CIFAR10H) better than previous
Now: ((image / hard label -> image / soft label) -> classification) 

* We basically try to see if we can predict human uncertainty, because having a dataset with a distribution of uncertainty can make it more robust.

* Explore model architecture optimization, with diff CNN architectures

* Ensemble learning with different architectures

* ALL transfer learning and fine tuning on CIFAR-10 data

* Qualitative analysis on when soft labels differ from ground truth

Motivation: What problem are you tackling, and what's the setting you're considering?

Method: What machine learning techniques have you tried and why?

Preliminary experiments: Describe the experiments that you've run, the outcomes, and any error analysis that you've done. You should have tried at least one baseline.

* We know at this point that using soft labels, even if they do not exactly match the hard labels, can improve the performance and robustness of the model.

Next steps: Given your preliminary results, what are the next steps that you're considering?
