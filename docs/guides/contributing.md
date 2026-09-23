# Contributing

How work gets into Wave Forge, what "done" means, and how the docs are kept. For *what* we are
building and *why*, read [vision.md](../product/vision.md) and the
[architecture overview](../architecture/overview.md) first. To set up a machine, read
[environment.md](environment.md); to run the tests, [testing.md](testing.md).

## Workflow

1. Branch from `develop`, in a git worktree if you work on several things at once. Build a worktree
   with `CARGO_TARGET_DIR` outside the checkout ([environment.md](environment.md)).
2. Commit atomically: one logical change per commit, and every commit builds and passes the tests.
3. Run the formatter, Clippy and the tests of every workspace the change touches before pushing
   ([testing.md](testing.md)).
4. Push and open a pull request against `develop`, never `main`, naming the issues it covers.
5. Wait for CI to be green and fix what fails before merging. CI skips a pull request that changes
   only Markdown, `docs/` or `LICENSE` ([environment.md](environment.md)), so check such a change
   by reading it.
6. Merge.
7. Close the issues the work resolved by hand. "Closes #n" in a commit or a pull request only closes
   an issue when it lands on the default branch, and `develop` is not the default branch. Comment on
   an issue that still has work in it instead of closing it.
8. Delete the merged branch locally and on `origin`, and remove the worktree. Never delete `develop`
   or `main`.

## The rules that hold for people and agents alike

- **The user stories are the done gate.** Nothing is considered done or published until every story
  in [user-stories.md](../product/user-stories.md) is verified by repeated, recorded checks with the
  evidence linked from it. Design and implementation name the stories they serve, and a change that
  affects a story updates its status and evidence.
- **Nothing from a private game repository enters this one.** The proof-of-concept games are
  closed source ([roadmap.md](../plan/roadmap.md)); their code, assets, design text and packs are
  never copied into this repository, its issues or its commits. A bug found in a game is reported
  with a reproduction in Wave Forge's own terms, using the
  [issue template](../../.github/ISSUE_TEMPLATE/from-a-game.md).

## Publishing is human-only

Releasing to the Godot Asset Store or crates.io, creating a release or a release tag, announcing,
and promoting `develop` to `main` are done by the owner, and so is anything like them. Agents prepare
builds, notes and checklists, and stop there.

## Documentation and comments

Runtime performance is our first priority, so parts of the code will be less obvious than the
simplest possible implementation (static dispatch, GPU kernels, packed data layouts, SIMD). That
trade-off is only acceptable if the reasoning is written down.

- **Explain why, not how.** The code shows what it does; comments and docs explain why it is done
  this way, which alternatives were rejected, and what measurement justified it.
- **Big picture in `docs/`, details next to the code.** Design and cross-cutting decisions go in
  `docs/architecture/`; anything local to one module belongs in comments in that module.
- **One home per topic.** Every topic has one document that owns it, and other documents link to
  that document rather than restating it. [docs/README.md](../README.md) lists which document owns
  what.
- **Each folder has one kind of content.**
  - `architecture/` is the design and why.
  - `reference/` is what is built, and is updated in the same pull request as the code it
    describes.
  - `plan/` is status and order of work. A change that fixes or introduces a known limit updates
    [status.md](../plan/status.md) in the same pull request.
  - `research/` is evidence.
- **Numbers live with their protocol.** Every measurement is recorded once, in
  [measurements.md](../research/measurements.md), with the build, machine, driver stack and command
  that produced it. Other documents link to it rather than copy the number. A performance claim in
  code or docs names the measurement it rests on.
- **Code cites docs by section name, not number.** A comment names the file and the section's
  heading, so a renumbered section does not leave a wrong pointer behind.
- **No em dashes, and plain complete sentences.** Write the current state; the history of a change
  belongs in its commit message and pull request.
