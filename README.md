# Bitstream hackathon website

This repo hosts the UW-PHARM group's bitstream hackathon website.

To develop this website, clone the repo, change directories to the root folder, and execute:
```shell
$ julia --project=.
```
This will launch Julia using the repo's root directory as the project environment. You will need to instantiate the project environment (and the tutorial environment) first. At the newly opened Julia REPL, press `]` to enter Pkg mode. Then run:
```julia-repl
(bitstream-hackathon) pkg> instantiate
# ... wait for installation to pass

(bitstream-hackathon) pkg> activate _tutorials
  Activating project at `~/Code/bitstream-hackathon/_tutorials`

(_tutorials) pkg> instantiate
# ... wait for installation to pass

(_tutorials) pkg> activate .
  Activating project at `~/Code/bitstream-hackathon`
```
Finally, you can run a local server with the website by exiting Pkg mode (press "backspace"), then running:
```julia-repl
julia> include("run.jl")

julia> preview()
```
After the website is finished building, a `localhost` server should open up.

## Adding tutorials

All the tutorials are written with [Literate.jl](https://github.com/fredrikekre/Literate.jl). This allows anyone to run the tutorials as Julia scripts. Adding a tutorials corresponds to:
1. Writing the Literate.jl tutorial under the `_tutorials` folder
2. Adding a markdown page for the tutorial under `tutorial`
3. Adding the page to the sidebar under `_layout/pgwrap.html`

You can see the other tutorials in these locations for an example how-to. Make sure you prepend your paths with `_tutorials` as the website builder will run the code from the repo root directory. When the tutorials are packaged for the participants, `_tutorials` is replaced by `.`.
