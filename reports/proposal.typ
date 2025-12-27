#set par(
  justify: true
)

= Matching Expressive Power: Comparative Study of Kolmogorov–Arnold Networks and MLPs - Proposal

Authors:
- Katrine Bukh Villesen (katrine.bukh.villesen\@estudiantat.upc.edu)
- Daniel Reverter Condal (daniel.reverter\@estudiantat.upc.edu)
- Marc Parcerisa Conesa (marc.parcerisa\@estudiantat.upc.edu)

== Problem Description

Multilayer Perceptrons (MLPs) with a single hidden layer are a well-understood baseline for studying expressive capacity and scaling behavior in neural networks. Recently, Kolmogorov-Arnold Networks (KANs) have been proposed as an alternative architecture that replaces fixed activation functions with learnable univariate functions, potentially offering improved expressive efficiency. Despite growing interest, there seems to be a lack of systematic understanding of how the expressive power of shallow KANs compares to that of shallow MLPs in terms of neuron efficiency. In particular, it remains unclear how many neurons a one-hidden-layer KAN requires to achieve performance comparable to a one-hidden-layer MLP with a given number of neurons, and how this relationship scales with model size and data availability. 

This study aims to address this gap through an extensive empirical comparison, quantifying the neuron-efficiency relationship between shallow KANs and MLPs across varying model widths and, if time permits, dataset sizes.

== Motivation

KANs have recently been proposed as an alternative to standard neural architectures and have attracted attention due to their potential expressive efficiency. Although KANs were supposed to be briefly discussed in this course, time constraints prevented a their exploration in class. Given the freedom to choose the project topic, this study was motivated by the opportunity to investigate a relatively novel and underexplored architecture. In particular, comparing KANs against well-established shallow MLPs provides a concrete and educational setting to analyze architectural differences in terms of capacity, scaling, and empirical performance.

== Techniques to be Used

The study will be conducted through systematic empirical evaluation implemented in Python. Both MLPs and one-hidden-layer KANs will be implemented using existing machine learning libraries where available, or custom implementations when necessary. For each architecture, the number of neurons in the single hidden layer will be varied to assess performance scaling and neuron efficiency. If time permits, additional experiments will explore the effect of varying the amount of training data. Cross-validation will be employed not for hyperparameter optimization, but to mitigate the effects of random weight initialization and provide statistically meaningful comparisons between architectures; and no train/test split will be performed, as the objective is just to evaluate the effect of said hyperparameters. Model performance will be evaluated using standard classification metrics.

== Data Source

Experiments will be performed on the Higgs Boson ATLAS Challenge dataset (2014), as per the previous two projects in this course.

== Computational Capabilities

All experiments will be run on local machines, primarily using CPUs, but we have access to a 12GB RTX 3060 GPU. Given the relatively small size of the dataset and the shallow architectures under consideration, local computational resources should be sufficient to conduct the experiments within a reasonable timeframe.

== Preliminary References

=== Foundational Reference:

- Ziming Liu and Yixuan Wang and Sachin Vaidya and Fabian Ruehle and James Halverson and Marin Soljačić and Thomas Y. Hou and Max Tegmark (2024). KAN: Kolmogorov-Arnold Networks. https://arxiv.org/abs/2404.19756

=== Additional Relevant References, provided by Gemini:
- Runpeng Yu and Weihao Yu and Xinchao Wang (2024). KAN or MLP: A Fairer Comparison. https://arxiv.org/abs/2407.16674
- Yunhong He and Yifeng Xie and Zhengqing Yuan and Lichao Sun (2024). MLP-KAN: Unifying Deep Representation and Function Learning. https://arxiv.org/abs/2410.03027