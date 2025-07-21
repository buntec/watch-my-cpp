# Watch My Cpp :eyes:

To launch the app, install [pixi](https://pixi.sh), then, from the root of this repo, do

```
./watch-my-cpp.py path/to/compile_commands.json
```

Finally, navigate to `localhost:8000` in your browser.

See `./watch-my-cpp.py -h` for all available options.

All sources and include directories are continuously watched for changes.
A change in a source file triggers a recompilation of that source.
A change in a header file triggers the recompilation of all sources
that have that header in their include paths (regardless of whether they
actually `#include` it). We use `ccache` to cache compilations.
See the ccache [docs](https://ccache.dev/manual/4.11.3.html#_configuration)
for configuration options.

Works with Clang and GCC on macOS and Linux. Tested on Firefox and Chrome.

![Screenshot on Firefox/Gnome](/screenshots/screenshot-firefox-gnome.png?raw=true)
