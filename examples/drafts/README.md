# Draft examples

`.ferx` files in this directory show **proposed syntax** for features whose
parser support has not yet landed. The parser smoke test in
`src/parser/model_parser.rs` skips files that declare blocks gated behind
unbuilt features, so running `cargo run -- examples/drafts/<file>.ferx`
today will fail.

The point is to give reviewers something concrete to comment on while the
larger feature is being staged across multiple PRs. When the corresponding
parser hookup lands, the file is moved up to `examples/` and becomes a
real, runnable example.

Currently empty — the previous occupant (`warfarin_dcm.ferx`) was promoted
to `examples/warfarin_dcm.ferx` once `[covariate_nn]` parsing + dispatch +
mu-ref recognition + tv_fn parity all landed. It still requires building
with `--features nn`. See `docs/model-file/covariate-nn.qmd` for usage.
