@echo off
echo Running tests for core packages...
echo Testing wfc-core...
cargo test -p wfc-core
echo Testing wfc-rules...
cargo test -p wfc-rules 