const { StaticPool } = require("node-worker-threads-pool");
const JSZip = require("jszip");
const axios = require("axios");
const util = require("util");

const commonArgs = [
    { name: "help", alias: "h", type: Boolean },
    {
        name: "url", alias: "u", type: String, defaultValue: "https://logs.tf/logs/log_%s.log.zip",
        description: "The main url path to fetch the log from. Default: https://logs.tf/logs/log_%s.log.zip"
    },
    {
        name: "steam64", alias: "s", type: Boolean, defaultValue: false,
        description: "Use steam64 ids instead of Steam3"
    },
    {
        name: "workers", alias: "w", defaultValue: 1, type: Number,
        description: "The amount of workers that should be used to parse logs"
    },
    {
        name: "log_format", alias: "l", type: Boolean,
        description: "Outputs the same format logs.tf uses"
    }
];

// Creates a pool to allow for threaded parsing
function createPool(workers, format, steam64) {
    return new StaticPool({
        size: workers,
        task: __dirname + "/process_lines.js",
        workerData: {
            format: format,
            steam64: steam64
        }
    });
}

function unwrapMap(map) {
    return Object.fromEntries(Array.from(
        map.entries()
        , ([k, v]) =>
            v instanceof Map
                ? [k, unwrapMap(v)]
                : [k, v]
    ));
}

// Turns nested maps into valid json objects
function jsonifyData(data) {
    data.PvC = unwrapMap(data.PvC);
    data.PvP = unwrapMap(data.PvP);
    data.playerClasses = unwrapMap(data.playerClasses)
    for (const player of Object.values(data.playerClasses)) {
        for (const [className, classData] of Object.entries(player)) {
            if (classData.weapons)
                classData.weapons = unwrapMap(classData.weapons);
            player[className] = classData;
        }
    }
    return data;
}

// Fetches a logfile from url and parses it
async function fetchLog(url, logid) {
    try {
        const response = await axios.get(util.format(url, logid), { responseType: "arraybuffer" });
        const zipData = await JSZip.loadAsync(response.data);
        const logFile = zipData.file(`log_${logid}.log`);
        if (logFile) {
            const log = await logFile.async("text");
            const logLines = log.split("\n");
            return logLines;
        }
    } catch (error) {
        console.error(`Failed to fetch log with id ${logid}. Reason:\n `, error);
    }
    return [];
}

// Adds a log to be processed by the pool
async function pushLog(pool, lines) {
    return pool.exec(lines);
}

exports.commonArgs = commonArgs;
exports.pushLog = pushLog;
exports.jsonifyData = jsonifyData;
exports.fetchLog = fetchLog;
exports.createPool = createPool;
