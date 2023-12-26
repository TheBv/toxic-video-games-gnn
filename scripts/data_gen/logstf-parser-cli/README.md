# Log parser CLI
A Command Line Interface to parse logs from Team Fortress 2 in various ways.

# Stand-alone
One can parse files and folders containing logs as well as a list of logids using `parse-log` an output folder can be specified using the `output -o` command line argument.

# Server
One can setup a server using `start-server` which allows one to either parse an entire file directly via the `/parse/` route or by id via the `id/:logid` route. Both return a json response.

# Format
By default the default format from the npm package `logstf-parser` is used. One can switch to the format logs.tf uses with the argument `log_format -l`

# Workers
To allow for faster parsing of logs one specify the amount of workers to be used when parsing logs with the argument `workers -w`. By default `1` worker is used.