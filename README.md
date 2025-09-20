# Watch My Cpp :eyes:

Using [uv](https://github.com/astral-sh/uv):

```sh
uvx git+https://github.com/buntec/watch-my-cpp --help

```

Prerequisites:

- Clang/GCC based C++ project with a `compile_commands.json`
- [ccache](https://ccache.dev)
- [clang-tidy](https://clang.llvm.org/extra/clang-tidy) (optional)
- [cppcheck](https://www.cppcheck.com) (optional)
- [include-what-you-use](https://github.com/include-what-you-use/include-what-you-use) (optional)

All sources and include directories are continuously watched for changes.
(Use `--ignore-patterns` to filter out parts of a project you don't want to watch/recompile.)
A change in a source file triggers a recompilation of that source.
A change in a header file triggers the recompilation of all sources whose include path contains that header (regardless of whether they actually `#include` it).
We use `ccache` to cache compilations.
See the ccache [docs](https://ccache.dev/manual/4.11.3.html#_configuration) for configuration options.

Works with on macOS and Linux. Tested on Firefox and Chrome.

![Screenshot on Firefox/Gnome](/screenshots/screenshot-firefox-gnome.png?raw=true)

# Dev

Prerequisites:

- [uv](https://github.com/astral-sh/uv)

Optional but recommended:

- [just](https://github.com/casey/just)
- [direnv](https://direnv.net/)
- [pixi](https://pixi.sh/latest/)
