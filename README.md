# Watch My Cpp :eyes:

# Use

Using [uv](https://github.com/astral-sh/uv):

```sh
uvx git+https://github.com/buntec/watch-my-cpp --help

```

All sources and include directories are continuously watched for changes.
A change in a source file triggers a recompilation of that source.
A change in a header file triggers the recompilation of all sources
that have that header in their include paths (regardless of whether they
actually `#include` it). We use `ccache` to cache compilations.
See the ccache [docs](https://ccache.dev/manual/4.11.3.html#_configuration)
for configuration options.

Works with Clang and GCC on macOS and Linux. Tested on Firefox and Chrome.

![Screenshot on Firefox/Gnome](/screenshots/screenshot-firefox-gnome.png?raw=true)

# Dev

Prerequisites:

- [bun](https://bun.com/)
- [uv](https://github.com/astral-sh/uv)

Optional but recommended:

- [just](https://github.com/casey/just)
- [direnv](https://direnv.net/)
- [pixi](https://pixi.sh/latest/)
