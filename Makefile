# The commands for developing this repo; `make help` lists them.
#
# setup.* is what this repo depends on, run once or after a pin bump. build.* is this repo's code, run after each edit.
# A dot means one level down, and clean.x undoes setup.x or build.x. Lines that need env.sh or the venv live in
# scripts/setup.sh or scripts/build.sh, as a function named after the part behind the dot; the rest stays here.
#
# After changing the compilers in env.sh, run make clean clean.triton. A moved or copied repo still points at the old
# folder, and a copy builds the old sources with no error, so also run make clean.venv, then make setup.

# ----------------------------------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------------------------------

.DEFAULT_GOAL := build
# A failed first install must rerun next time, or the package stays uninstalled while Ninja succeeds.
.DELETE_ON_ERROR:

# LLVM and LLVM_SOURCE from the make command line only. A variable of that name in the shell means something else.
llvm_args = "$(if $(filter command line,$(origin LLVM)),$(LLVM))" "$(if $(filter command line,$(origin LLVM_SOURCE)),$(LLVM_SOURCE))"

# ----------------------------------------------------------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------------------------------------------------------

# Each dot in a name indents it one level.
.PHONY: help
help: ## list these commands
	@awk -F':.*## ' '/^[a-z.]+:.*## /{d = gsub(/\./, ".", $$1); printf "  %*s%-*s %s\n", 2 * d, "", 20 - 2 * d, $$1, $$2}' \
		$(lastword $(MAKEFILE_LIST))

# ----------------------------------------------------------------------------------------------------------------------
# Setup: what this repo depends on
# ----------------------------------------------------------------------------------------------------------------------

# Each + hands Make's job slots to the Ninja inside, so make -j<n> limits it. Plain make already uses every core.
.PHONY: setup
setup: setup.atlas setup.hooks setup.llvm ## take a fresh clone to a working plugin; LLVM=<dir> or LLVM_SOURCE=1 picks the LLVM
	+$(MAKE) build
	@scripts/setup.sh llvm_check

.PHONY: setup.full
setup.full: setup ## setup, plus Triton and all of atlas-mlir: triton-opt, atlas-opt and the rest
	+$(MAKE) setup.triton setup.atlas.full
	+$(MAKE) build

.PHONY: setup.env
setup.env: env.sh ## create env.sh from the example

env.sh:
	cp env.example.sh $@

# Every step here that runs git comes after this one, because git locks .git/config and make -j would collide on it.
.PHONY: setup.atlas
setup.atlas: ## fetch atlas-mlir; the plugin's build compiles it
	git submodule update --init third_party/atlas-mlir

.PHONY: setup.atlas.full
setup.atlas.full: setup.llvm ## build all of atlas-mlir: atlas-opt, atlas-emit and its tests (1 min)
	+scripts/setup.sh atlas_full
	+scripts/build.sh tools

.PHONY: setup.hooks
setup.hooks: setup.atlas ## turn on the git hooks
	git config core.hooksPath .githooks

.PHONY: setup.venv
setup.venv: env.sh ## create .venv
	scripts/setup.sh venv

.PHONY: setup.llvm
setup.llvm: setup.atlas setup.venv ## get LLVM: downloads one unless LLVM=<dir> or LLVM_SOURCE=1 says otherwise
	+scripts/setup.sh llvm $(llvm_args)

.PHONY: setup.llvm.prebuilt
setup.llvm.prebuilt: setup.venv ## download Triton's LLVM (2 GB); it has no mlir-opt or lld
	scripts/setup.sh llvm_prebuilt

.PHONY: setup.llvm.source
setup.llvm.source: setup.venv ## build LLVM from the submodule, with all its tools (22 min, 9 GB)
	+scripts/setup.sh llvm_source

.PHONY: setup.llvm.check
setup.llvm.check: ## report what the LLVM in use lacks
	@scripts/setup.sh llvm_check

.PHONY: setup.triton
setup.triton: setup.llvm ## fetch and compile Triton (4 min, 8 GB); make then builds the full server
	+scripts/setup.sh triton
	+scripts/build.sh tools

.venv/pyvenv.cfg .venv/bin/pre-commit:
	@echo "error: $@ missing; run make setup" >&2; exit 1

# ----------------------------------------------------------------------------------------------------------------------
# Build: this repo's code
# ----------------------------------------------------------------------------------------------------------------------

.PHONY: build
build: build.plugin build.lsp build.tools ## build the plugin and the language servers (default)

# Reinstalling takes time even with nothing to do, so it reruns only when what it reads changes.
build/plugin/build.ninja: env.sh pyproject.toml uv.lock .venv/pyvenv.cfg
	+scripts/build.sh package
	touch $@

.PHONY: build.plugin
build.plugin: build/plugin/build.ninja ## rebuild the plugin and put it in the venv
	+scripts/build.sh plugin

# The small server shares build/plugin with the plugin, and two Ninjas can't share a folder.
.PHONY: build.lsp
build.lsp: build.plugin ## the language server: full if Triton is built, otherwise small
	+scripts/build.sh lsp

.PHONY: build.lsp.small
build.lsp.small: build.plugin ## the server linked to the wheel
	+scripts/build.sh lsp_small

.PHONY: build.lsp.full
build.lsp.full: setup.triton ## the server built from Triton's source
	+scripts/build.sh lsp_full

.PHONY: build.lsp.tblgen
build.lsp.tblgen: ## the server for .td files
	+scripts/build.sh lsp_tblgen

.PHONY: build.tools
build.tools: build.plugin ## keep atlas-mlir current if built, and link every tool in build/bin
	+scripts/build.sh tools

.PHONY: build.wheel
build.wheel: build/plugin/build.ninja ## build the wheel to hand out, in build/dist
	+scripts/build.sh wheel

# ----------------------------------------------------------------------------------------------------------------------
# Format
# ----------------------------------------------------------------------------------------------------------------------

.PHONY: format
format: .venv/bin/pre-commit ## run the formatters and linters on every file
	.venv/bin/pre-commit run --all-files

# ----------------------------------------------------------------------------------------------------------------------
# Clean
# ----------------------------------------------------------------------------------------------------------------------

# These need neither env.sh nor the venv, so they still work when either is broken. clean keeps what is slow to rebuild.
.PHONY: clean
clean: clean.plugin clean.lsp ## delete the plugin and small server; keeps LLVM, Triton and the venv

# An editor's running language server can keep files open on NFS, so a partial clean isn't an error. build.ninja goes
# first, so whatever stays behind still gets reconfigured.
.PHONY: clean.plugin
clean.plugin: ## delete the plugin's build and the editor links
	-rm -f build/plugin/build.ninja
	-rm -rf build/plugin build/dist compile_commands.json build/tablegen_compile_commands.yml
	@[ ! -e build/plugin ] || echo "warning: build/plugin not fully removed; close the editor and rerun"

.PHONY: clean.lsp
clean.lsp: ## delete the links editors use and the small and .td servers
	-rm -f build/triton-atlas-lsp build/plugin/bin/triton-atlas-lsp build/tblgen-lsp-server
	-rm -rf build/tblgen-lsp

.PHONY: clean.triton
clean.triton: ## delete the compiled Triton, and with it the full server
	-rm -rf build/triton build/triton-cache build/bin

.PHONY: clean.atlas
clean.atlas: ## delete the full atlas-mlir build
	-rm -rf build/atlas build/bin

.PHONY: clean.llvm
clean.llvm: ## delete the LLVM in build/llvm
	rm -rf build/llvm

.PHONY: clean.venv
clean.venv: ## delete .venv
	rm -rf .venv
