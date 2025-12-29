#set document(title: [Climate Downscaling via spectral Autoregressive Transformer Modelling])

#import "@preview/subpar:0.2.2"

#set text(
  size: 12pt,
  font: "New Computer Modern",
)

//#show raw: set text(font: "New Computer Modern Mono")

#show heading: set block(above: 1.4em, below: 1em)
#set heading(numbering: "1.")

#show title: set text(size: 18pt)
#show title: set align(center)
#show title: set block(above: 3em, below: 3em)

// Title page
#page(
  paper: "a4",
  header: none,
  footer: none,
  numbering: none,
)[

#title()

A research project conducted by #link("mailto:luis.gindorf@student.uni-tuebinegen.de", "Luis Gindorf")

#v(1em)

#grid(
  columns: (1fr, 1fr),
align(center)[
Direct Supervision:\
Jannik Thümmel\
],
align(center)[
Managing Supervision:\
Junior Professor Nicole Ludwig\
])
#align(center)[
Climate, Energy and Machine Learning Systems Group\
Tübingen AI Center, Eberhard-Karls-Universität Tübingen]
#v(2em)

#align(center)[
  #set par(justify: false)
  #set text(size: 15pt)
  *Abstract* \
]
#lorem(80)
]


#set page(
  paper: "a4",
  header: align(right + horizon, context document.title),
  numbering: "1",
)

#set heading(
  numbering: "1.",
)

// Reduce spcing below headings
#show heading: set block(below: 0.5em)

#outline(title: "Outline", depth: 1)

#heading("Introduction")
Climate change affects humanity not only through global and long time-scale effects such as loss of land due to rising sea levels @Climate-Change-Effects-1 and hotter summers @Climate-Change-Effects-2, but also through more local phenomena including but not limited to growing frequency of heat waves and extreme precipitation events as well as more extreme cyclones @Climate-Change-Effects-3 @Climate-Change-Effects-4. While these changes can already be observed, it is paramount for informed policy-making to forecast the frequency and severity of these extreme weather events. Earth system model simulations enable long time forecasting of environment variables under climate change, however, due to their computational complexity, are limited to spatial resolutions of typically 100-300 kilometers @Overview-ESMs.
These are too coarse to enable analysis of local extreme weather phenomena. To close this gap, recently, machine learning models have been proposed to downscale earth system model predictions to higher spatial resolutions @downscaling-geospatial-attention @downscaling-diffusion-1 @downscaling-diffusion-2 @downscaling-normalizing-flow @downscaling-GANs @downscaling-probabilistic . This work applies a novel approach to do exactly this: The data is transformed into the spectral domain, in order to predict higher-frequency components autoregressively. Retransformation then yields an upscaled version of the input. The concrete methodology is described in more detail in @methods.
In @results, the models performance is shown and compared to a linear interpolation baseline. A comparison to a state of the art diffusion model for climate downscaling would make sense, but is outside the scope of this research project. 

= Methodology <methods>
What follows in this section is a thorough description of the data processing pipeline underlying this project. 

== Data
The dataset stems from the ERA5 Copernicus Climate Change Service Climate Data Store @data. Only the wind speed data at a level of $500$ mBar was used. This results in a data corpus with one datapoint every  six hours from 1959 till 2022. Every datapoint consists of an equiangular grid of $N_"lat" times N_"lon" = 121 times 240$. The data is divided into the following split by years:

#table(
  columns: 4,
  [*Split*], [Train], [Validation], [Test],
  [*Years*], [1959 - 2005], [2006 - 2012], [2013 - 2022],
  [Datapoints], [68664], [10228], [13148],
  [Percentage], [75%], [11%], [14%])

More than 10.000 samples in validation and test set should ensure sufficiently small variances in the reported results. The choice to slice the validation and test set chronologically and not via random subsampling was made deliberately to evaluate the model on a slightly shifted distribution away from the one used for training (due to climate change). This reflects the use case of training on past data and evaluating the model on future (forecast) data more closely.

== Spherical Harmonics transformation
The main preprocessing step performed is the transformation into the spectral domain. Since the data lies on an equiangularly gridded sphere, this is done using the spherical harmoncis transformation, which is implemented in @torch-harmonics. For a datapoint of shape $(121, 240)$, this results in a lower-triangular matrix of shape $(121,121)$ with complex entries. These are flattened row-by-row. The sequence length is $T = "triangular_number"(121) = (121 times 122) / 2$ = 7381. Finally, they are viewed as real, to obtain a sequence of real-valued two-dimensional vectors. Lower indices in the sequence represent lower-frequency features in the data.

== Autoregressive Probabilistic Modelling
This fact can be exploited to define a learning problem where higher-order coefficients (so, later vectors in the sequence) are predicted from lower-order ones. We define $L = "triangular_number"(61)= 1891$ as the length of the input sequence. This corresponds to an input image of resolution (61, 120), so half of the orginal resolution. 
Given the vectors in the input sequence, the model autoregressively predicts the next elements of the sequence until index $T$ is reached. 
For this kind of autoregressive sequence modelling with lots of training data, the choice of a transformer model is natural. In order to enable the prediction of real-valued vectors, we model them as Gaussian mixture models (GMMs), which the model parameterizes. 
In order to do so, the model has a head which coerces the hidden state for a sequence element of the transformer to 
 - normalized GMM mixture weights using the softmax
 - GMM component means
 - isotropic GMM component variances using the softplus and thresholding to $10^(-5)$
 analogous to @GMM-Transformer. The number of GMM components is fixed at eight.
 During inference, the predicted distribution parameters are used to sample a sequence element. During training, the negatve log likelihood (NLL) of the true sequence elements under the distribution parameterized by the network is used as a loss function. To prevent common pitfalls in training of parameter estimation with NLL, $beta$-NLL is employed with a parameter of $beta = 0.5$. @beta-NLL

== Transformer Model
A standard decoder-only transformer model is used with a hidden dimension of 512 and four sequential transformer blocks with eight attention heads each. The learning rate follows a cosine annealing decay after a linear warmup of one epoch with a maximum of $5 times 10^(-5)$. For regularization, early stopping on the NLL loss of the validation step and dropout with probability $10%$ is used. 
These details follow @diffusion-transformer and @attention along with some manual tuning.


= Results <results>


#subpar.grid(
  figure(image("Figures/Ground-Truth_Sample.pdf", width: 105%), caption: [Ground truth sample]), <a>,
  figure(image("Figures/Inference-Input_Sample.pdf"), caption: [Inference input]), <b>,
  figure(image("Figures/Prediction_Sample.pdf", width: 100%), caption: [Model output]), <c>,
  figure(image("Figures/Prediction-Difference_Sample.pdf"), caption: [Difference between @a and @c]), <d>,
  figure(image("Figures/Colorbar.pdf", width: 100%)),
  columns: (4fr, 4fr, 0.5fr),
  rows: (1fr, 1fr),
  caption: [Ground truth (@a) and inference model input (@b) from the first sample of the testset.],
  label: <sample-visualizations>,
)

= Conclusion <conclusion>

#pagebreak()

#bibliography("bibliography.yaml", style: "ieee.csl")

