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

* Splitting stuff:
    * Model baselines: Jack
    * Find interesting patterns: Alex
    * Writing: Connor
        * Maybe get a sense of what people are doing and thinking of extensions/interesting methodologies

* Githubs
    * alex-gugu
    * cdhmarsh


# Milestone 2 (after the milestone submission)
* Identify some failure cases of the best baseline model
* Try out some potential solutions to improve the model performance
* The improvement can be in the aspect of model, dataset, training method, loss function, etc. 


# Final Goal
When I say taking "model architecture optimization, ensemble methods, or cross-dataset generalization" as the final goal of the project, I'm actually citing your last sentence of "Intended experiments". I think besides taking the CIFAR-10H dataset and training an existing model with that data, you can consider ways to improve the model architecture or try to ensemble different models/methods to increase the classification accuracy, or apply similar ideas to other datasets/applications. These are some ways to expand the scope of your project.