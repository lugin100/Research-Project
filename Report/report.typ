#set document(title: [Climate Downcasting via Frequency-Domain Autoregressive Transformer Modelling])


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
  numbering: "1."
)

#outline(title: "Outline")

#heading("Introduction")
Climate change affects humanity not only through global and long time-scale effects such as loss of land due to rising sea levels @Climate-Change-Effects-1 and hotter summers @Climate-Change-Effects-2, but also through more local phenomena including but not limited to growing frequency of heat waves and extreme precipitation events as well as more extreme cyclones @Climate-Change-Effects-3 @Climate-Change-Effects-4. While these changes can already be observed, it is paramount for informed policy-making to forecast the frequency and severity of these extreme weather events. Earth system model simulations enable long time forecasting of environment variables under climate change, however, due to their computational complexity, are limited to spatial resolutions of typically 100-300 kilometers @Overview-ESMs.
These are too coarse to enable analysis of local extreme weather phenomena. To close this gap, recently, machine learning models have been proposed to downscale earth system model predictions to higher spatial resolutions @downscaling-geospatial-attention @downscaling-diffusion-1 @downscaling-diffusion-2 @downscaling-normalizing-flow @downscaling-GANs @downscaling-probabilistic .



#heading("Methodology")


#heading("Results")

#heading("Conclusion")

#pagebreak()

#bibliography("bibliography.yaml", style: "ieee.csl")

