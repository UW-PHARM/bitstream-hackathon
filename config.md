<!--
Add here global page variables to use throughout your website.
-->
+++
author = "UW-Madison PHARM Group"
mintoclevel = 2

# Add here files or directories that should be ignored by Franklin, otherwise
# these files might be copied and, if markdown, processed by Franklin which
# you might not want. Indicate directories by ending the name with a `/`.
# Base files such as LICENSE.md and README.md are ignored by default.
ignore = ["node_modules/"]

# URL prepath
prepath = "bitstream-hackathon"
+++

<!--
Add here global latex commands to use throughout your pages.
-->
\newcommand{\tutorial}[1]{
  *Make sure you have completed the [getting started](/tutorials/overview) tutorial.*

  **Table of contents:**
  
  \toc\literate{/_tutorials/!#1}
}
