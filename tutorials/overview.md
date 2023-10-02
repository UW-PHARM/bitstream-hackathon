@def title = "Tutorials (overview)"

# Getting started with tutorials

This page will guide you through setting up your Julia environment and the tutorials folder. **Make sure your complete this guide before moving on to any other tutorials!**

\toc

## Setting up Julia and VS Code

Going from nothing to a complete Julia IDE is easy. Just follow these steps:

1. Install Julia by downloading v1.7.2 [here](https://julialang.org/downloads/) and installing the binary as per your system requirements. Ensure that your installation is working by opening your terminal and running (if this doesn't work for you, then you may need to add the `julia` binary to your system path):
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

2. Download [VS Code](https://code.visualstudio.com) (the preferred Julia editor). You can use your own editor, but then you will not be able to take advantage of the Julia integration with VS Code. In this case, you would need to copy-and-paste code snippets into a Julia REPL on your terminal.

3. Install the [Julia extension for VS Code](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia). Test the extension by opening the VS Code command palette and running "Julia: Start REPL". You should see the same Julia prompt as above show up in a VS Code terminal.

## Setting up the hackathon tutorials

Download the all hackathon tutorials files [here](/assets/tutorials.tar.gz). Uncompress and save them under a shared folder.

Open the folder you just saved in VS Code. You should see the Julia language server autodetect the tutorial environment, and the bottom pane of VS Code will read `Julia env: <name of your folder>`.

Install all the packages by opening a new REPL via VS Code (see the [previous section](#setting-up-julia-and-vs-code)). Press the `]` key to enter Pkg mode. You should see the name of your tutorial folder at the prompt (this is your current Pkg "environment"). Run the command `instantiate` to install all the packages required for the tutorials.

Now, you can run the tutorials! Hit the `backspace` key to exit Pkg mode, and run
```julia-repl
julia> include("<tutorial file>")
```

*NOTE:* if you open the tutorial folder in VS Code as described above, then running "Julia: Start REPL" in VS Code will start a new REPL with the tutorial folder environment already activated. You can always manually specify the environment when you launch Julia from the terminal via
```bash
$ julia --project=.
```
where the `.` denotes the "current working directory" as the path for the project. You can [read the Pkg docs](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project) for more details.

## Accessing the pre-trained model and dataset

We provide all participants with a pre-trained version of MobileNet-v1 designed to be used with bitstream computing. This model matches the original network with a width multipler of 0.25, input resolution of 96 by 96, and an activation function of hardtanh. You can access the pre-trained model by setting up the tutorial folder as described in [above](#setting-up-the-hackathon-tutorials). Then, in your code, add
```julia
# this is the setup file included in the tutorial
include("src/setup.jl")

BSON.@load "src/pretrained.bson" m
```

We also provide a version of the [Visual Wake Words dataset](https://arxiv.org/abs/1906.05721) via the artifacts system.

```julia
# this is the setup file included in the tutorial
include("src/setup.jl")

dataroot = joinpath(artifact"vww", "vww-hackathon")
# get the train split (options are :train, :test, :val)
traindata = VisualWakeWords(dataroot; subset = :train)
```
