#!/usr/bin/env node

const fs = require("fs");
const commandLineArgs = require("command-line-args");
const commandLineUsage = require("command-line-usage");
const path = require("path");
const { fetchLog, pushLog, createPool, jsonifyData, commonArgs } = require("./common")

const optionDefinitions = [
    ...commonArgs,
    {
        name: "file", alias: "f", multiple: true, type: String,
        description: "Files to be processed"
    },
    {
        name: "dir", alias: "d", type: String,
        description: "The path to a folder of log files to be parsed"
    },
    {
        name: "id", alias: "i", multiple: true, type: String,
        description: "The logids to process"
    },
    {
        name: "output", alias: "o", type: String,
        description: "Specifies the output folder. If none is specified it prints to console"
    }
]
const sections = [
    {
        header: "Log parser cli",
        content: "Options for parsing log files"
    },
    {
        header: "Options",
        optionList: optionDefinitions
    }
];
const usage = commandLineUsage(sections);
const options = commandLineArgs(optionDefinitions);

function main() {
    if (options.help) {
        console.log(usage);
        return;
    }
    const pool = createPool(options.workers, options.log_format, options.steam64)
    const promises = []
    if (options.output) {
        if (!fs.existsSync(`${options.output}`))
            fs.mkdirSync(`${options.output}`);
    }
    if (options.id) {
        for (const id of options.id) {
            addPromise(promises, async () => {
                console.log(options.url)
                const lines = await fetchLog(options.url, id);
                const res = await pushLog(pool, lines);
                writeOutput(res, id);
            });
        }
    }
    if (options.dir) {
        const dir = fs.readdirSync(options.dir);
        for (const file of dir) {
            if (file.endsWith(".log")) {
                addPromise(promises, async () => {
                    const path = options.dir + file;
                    const lines = fs.readFileSync(path, "UTF-8").split("\n");
                    const res = await pushLog(pool, lines);
                    writeOutput(res, getFileName(path));
                });
            }
        }
    }
    if (options.file) {
        for (const file of options.file) {
            addPromise(promises, async () => {
                const lines = fs.readFileSync(file, "UTF-8").split("\n");
                const res = await pushLog(pool, lines);
                writeOutput(res, getFileName(path));
            });
        }
    }
    Promise.all(promises).then(() => { pool.destroy(); });
}

function getFileName(fpath) {
    return path.basename(fpath, path.extname(fpath));
}

function addPromise(promises, func) {
    promises.push(func());
}

function writeOutput(jsonContent, fileName = undefined) {
    if (!options.log_format)
        jsonContent = jsonifyData(jsonContent);
    if (options.output) {
        fs.writeFileSync(`${options.output}/${fileName}.json`, JSON.stringify(jsonContent));
    }
    else {
        console.log(jsonContent);
    }
}

if (require.main === module) {
    main();
}
