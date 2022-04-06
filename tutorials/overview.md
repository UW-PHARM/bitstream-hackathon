@def title = "Tutorials (overview)"

# Getting started with tutorials

This page will guide you through setting up your Julia environment and the tutorials folder. **Make sure your complete this guide before moving on to any other tutorials!**

\toc

## Setting up Julia and VS Code

Going from nothing to a complete Julia IDE is easy. Just follow these steps:

1. Install Julia by downloading v1.7.2 [here](https://julialang.org/downloads/) and installing the binary as per your system requirements. Ensure that your installation is working by opening your terminal and running:
    ```txt
    $ julia
                   _
       _       _ _(_)_     |  Documentation: https://docs.julialang.org
      (_)     | (_) (_)    |
       _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
      | | | | | | |/ _` |  |
      | | |_| | | | (_| |  |  Version 1.7.2 (2022-02-06)
     _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
    |__/                   |
    
    julia>
    ```
    If this doesn't work for you, then you may need to add the `julia` binary to your system path.

2. Download [VS Code](https://code.visualstudio.com) (the preferred Julia editor). You can use your own editor, but then you will not be able to take advantage of the Julia integration with VS Code. In this case, you would need to copy-and-paste code snippets into a Julia REPL on your terminal.

3. Install the [Julia extension for VS Code](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia). Test the extension by opening the VS Code command palette and running "Julia: Start REPL". You should see the same Julia prompt as above show up in a VS Code terminal.

## Setting up the hackathon tutorials

Download the all hackathon tutorials files [here](https://github.com/UW-PHARM/bitstream-hackathon/tree/gh-pages/_tutorials). Make sure to grab _all the files_. Save them under a shared folder.

Open the folder you just saved in VS Code. You should see the Julia language server autodetect the tutorial environment, and the bottom pane of VS Code will read `Julia env: <name of your folder>`.

Select a tutorial that you would like to run. All the tutorials are stored under a folder called `_tutorials` on the website. You need to delete this prefix from all the paths in the tutorial. For example, if the tutorial contains `include("_tutorials/src/setup.jl")`, then you would change it to `include("src/setup.jl")`.

Now, you can run the tutorial! Open up a new REPL in VS Code, and run
```julia-repl
julia> include("<tutorial file>")
```
