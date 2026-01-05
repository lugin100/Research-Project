#set document(title: [Climate Downscaling via Spectral Autoregressive Transformer Modelling])

#import "@preview/subpar:0.2.2"  // For subplots

#set text(
  size: 12pt,
  font: "New Computer Modern",
)

//#show raw: set text(font: "New Computer Modern Mono")

#show heading: set block(above: 1.4em, below: 1em)
#set heading(numbering: "1.")

#show title: set text(size: 19pt)
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

#pad(left: 4cm, top: -1cm, figure(image("Figures/logo-excellence-cluster.png", width: 50%)))

#v(1fr)

#align(center)[
  #set par(justify: false)
  #set text(size: 15pt)
  *Abstract* \
]
#lorem(80)

#v(2fr)
#outline(title: "Outline", depth: 1)
]

#set page(
  paper: "a4",
  header: align(right + horizon, context document.title),
  numbering: "1",
  margin: (x: 2cm),
)

#set heading(
  numbering: "1.",
)
// Reduce spacing below headings
#show heading: set block(below: 0.5em)

#heading("Introduction")
Climate change affects humanity not only through global and long time-scale effects such as loss of land due to rising sea levels @Climate-Change-Effects-1 and hotter summers @Climate-Change-Effects-2, but also through more local phenomena including but not limited to growing frequency of heat waves and extreme precipitation events as well as more extreme cyclones @Climate-Change-Effects-3 @Climate-Change-Effects-4. While these changes can already be observed, it is paramount for informed policy-making to forecast the frequency and severity of these extreme weather events. Earth system model simulations enable long time forecasting of environment variables under climate change, however, due to their computational complexity, are limited to spatial resolutions of typically 100-300 kilometers @Overview-ESMs.
These are too coarse to enable analysis of local extreme weather phenomena. To close this gap, recently, machine learning models have been proposed to downscale earth system model predictions to higher spatial resolutions @downscaling-geospatial-attention @downscaling-diffusion-1 @downscaling-diffusion-2 @downscaling-normalizing-flow @downscaling-GANs @downscaling-probabilistic . This work applies a novel approach to do exactly this: The data is transformed into the spectral domain, in order to predict higher-frequency components autoregressively. Retransformation then yields an upscaled version of the input. The concrete methodology is described in more detail in @methods.
In @results, the models performance is shown and compared to a linear interpolation baseline. A comparison to a state of the art diffusion model for climate downscaling would make sense, but is outside the scope of this research project. 

= Methodology <methods>
What follows in this section is a thorough description of the data processing pipeline and modelling underlying this project. 

== Data
The dataset stems from the ERA5 Copernicus Climate Change Service Climate Data Store @data. Only the wind speed data at a level of $500$ mBar was used. This results in a data corpus with one datapoint every  six hours from 1959 till 2022. Every datapoint consists of an equiangular grid of $N_"lat" times N_"lon" = 121 times 240$. The data is divided into the following split by years:

#table(
  columns: 4,
  [*Split*], [Train], [Validation], [Test],
  [*Years*], [1959 - 2005], [2006 - 2012], [2013 - 2022],
  [*Samples*], [68664], [10228], [13148],
  [*Percentage*], [75%], [11%], [14%])

More than 10.000 samples in validation and test set should ensure sufficiently small variances in the reported results. The choice to slice the validation and test set chronologically and not via random subsampling was made deliberately to evaluate the model on a slightly shifted distribution away from the one used for training (due to climate change). This reflects the use case of training on past data and evaluating the model on future (forecast) data more closely.

== Spherical Harmonics transformation
The main preprocessing step performed is the transformation into the spectral domain. Since the data lies on an equiangularly gridded sphere, this is done using the spherical harmoncis (SH) transformation, which is implemented in @torch-harmonics. For a sample of shape $(121, 240)$, this results in a lower-triangular matrix of shape $(121,121)$ with complex entries. These are flattened row-by-row. The sequence length is $T = "triangular_number"(121)$$ = (121 times 122) / 2$ = 7381. Finally, they are viewed as real, to obtain a sequence of real-valued two-dimensional vectors. Lower indices in the sequence represent lower-frequency features in the data.

== Autoregressive Probabilistic Modelling
This fact can be exploited to define a learning problem where higher-order coefficients (so, later vectors in the sequence) are predicted from lower-order ones. The length of the input sequence is $L = "triangular_number"(61)= 1891$. This corresponds to an input image of resolution (61, 120), so half of the orginal resolution. 
Given the vectors in the input sequence, the model autoregressively predicts the next elements of the sequence until index $T$ is reached. 
For this kind of autoregressive sequence modelling with lots of training data, the choice of a transformer model is natural. In order to enable the prediction of real-valued vectors, we model them as Gaussian mixture models (GMMs), which the model parameterizes. 
In order to do so, the model has a head which coerces the hidden state for a sequence element of the transformer to 
 - normalized GMM mixture weights using the softmax
 - GMM component means
 - isotropic GMM component variances using the softplus and thresholding to $10^(-5)$
 analogous to @GMM-Transformer. The number of GMM components is fixed at eight.
 During inference, the predicted distribution parameters are used to sample a sequence element. During training, the negatve log likelihood (NLL) of the true sequence elements under the distribution parameterized by the network is used as a loss function. To prevent common pitfalls in training of parameter estimation with NLL, $beta$-corrected NLL is employed as loss function with $beta = 0.5$. @beta-NLL

== Transformer Model
A standard decoder-only transformer model is used with a hidden dimension of 512 and five sequential transformer blocks with eight attention heads each. The learning rate follows a cosine annealing decay after a linear warmup of 200 steps with a maximum of $5 times 10^(-5)$. For regularization, early stopping on the NLL loss of the validation step and dropout with probability $10%$ is used. 
These details follow @diffusion-transformer and @attention along with some manual tuning. Overall, the model has $23.3$
 million parameters. The model was optimized for 1344 steps with AdamW @AdamW and a batch size of 32 until the best checkpoint was reached.


= Results <results>
The transformer model achieves a root mean squared error (RMSE) on the testset of $1.84 m/s$. For comparison, bilinear interpolation achieves an RMSE of $2.10 m/s$. The predictions of the first testset sample are plotted in @sample-interpolation for the interpolation and @sample-model for the transformer model. They show that the interpolation makes errors predominantly in regions of large change of wind speed, while the transformer models errors appear more like small dots, likely stemming from spherical harmonic coefficients with large degree. @coefficient-RMSE shows the RMSE on those coefficients, averaged over the test set. This shows that the errors do not increase during autoregressive rollout, i.e. sampling errors do not accumulate. In @RMSE-over-time, the (smoothed) RMSE is plotted over the timestamps of the test set samples. This shows two things: The error does not seem to increase over time, which could have been caused by distribution shift away from the train set that ends in 2005. Secondly, it shows a seasonal trend that is weak in amplitude but consistent. Predictions are slightly better in (northern hemisphere) autumn and winter, and worse in spring/summer.
#pad(x: -2cm,
subpar.grid(
  figure(image("Figures/Ground-Truth_Sample.pdf", width: 105%), caption: [Ground truth]), <1a>,
  figure(image("Figures/Inference-Input_Sample.pdf"), caption: [Inference input]), <1b>,
  pad(bottom: 1.5cm, figure(image("Figures/Colorbar.pdf", height: 19%))),
  columns: (4fr, 4fr, 0.5fr),
  caption: [First sample from the test set.],
  label: <sample-ground-truth>,
))
#pad(x: -2cm,
subpar.grid(
  figure(image("Figures/Interpolation-Prediction_Sample.pdf", width: 100%), caption: [Interpolation output]),
  figure(image("Figures/Interpolation-Prediction-Difference_Sample.pdf"), caption: [Difference to ground truth]),
  pad(bottom: 1.5cm, figure(image("Figures/Interpolation-Difference-Colorbar.pdf", height: 19%))),
  columns: (4fr, 4fr, 0.5fr),
  caption: [Bilinear interpolation on the sample shown in @1b. Shows smoothed variant of the inference input. Errors are predominantly along edges of large change in wind speed.],
  label: <sample-interpolation>,
))
#pad(x: -2cm,
subpar.grid(
  figure(image("Figures/Model-Prediction_Sample.pdf", width: 100%), caption: [Model output]),
  figure(image("Figures/Model-Prediction-Difference_Sample.pdf"), caption: [Difference to ground truth]),
  pad(bottom: 1.5cm, figure(image("Figures/Model-Difference-Colorbar.pdf", height: 18%))),
  columns: (4fr, 4fr, 0.5fr),
  caption: [Model prediction on the sample shown in @1b. Shows errors are predominantly small-scale and larger near the poles.],
  label: <sample-model>,
))

#pad(x: -2cm,
subpar.grid(
  figure(image("Figures/Coefficient_RMSE.pdf", width: 100%), caption: [RMSE for SH coefficients. Coefficient matrix is lower-triangular and coefficients $<= 61$ are model input, hence whitespace in the figure that corresponds to no error.]), <coefficient-RMSE>,
  figure(image("Figures/RMSE_over_time.pdf"), caption: [RMSE over time on the test set. Values are smoothed by batch averaging over six days and then exponential moving averaging with smoothing factor $alpha = 0.05$.]), <RMSE-over-time>,
  columns: (1fr, 1fr),
))

= Conclusion <conclusion>

Promising results -> Probably error decreases with larger model (can learn weather distribution)
Stable rollout -> Models with higher resolution thinkable
No visible distribution shift -> Model seems robust enough to perform prediction on simulations
Next steps: Compare to state of the art diffusion models, train for other weather variables


#text(size: 14pt, weight: "bold")[Code Availability] #linebreak()
The code for this work is available on GitHub under #link("https://github.com/lugin100/Research-Project")
#pagebreak()

#bibliography("bibliography.yaml", style: "ieee.csl")

