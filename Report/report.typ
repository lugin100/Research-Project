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
CEALS Group\
Tübingen AI Center
],
align(center)[
Managing Supervision:\
Junior Professor Nicole Ludwig\
CEALS Group\
Tübingen AI Center
])

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



#heading("Methodology")


#heading("Results")

#heading("Conclusion")

#pagebreak()

#bibliography("bibliography.yaml", style: "ieee")

