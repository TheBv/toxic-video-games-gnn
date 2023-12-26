const defaultEvent = () => ({
    attacker: null,
    victim: null,
    headshot: false,
    airshot: 0,
    medicDrop: false,
    second: -1,
    capture: null,
    kill: false,
    backstab: false,
    medicDeath: false,
    advantageLost: null,
    chargeUsed: false,
    weapon: null,
    message: null
})

class EventsModule {

    constructor(gameState) {
        this.identifier = 'events'
        this.gameStartTime = null
        this.gameState = gameState
        this.mysqlEvents = []
        this.kills = new Map()
    }

    damageEventToMysql(event, mysql) {
        mysql.attacker = event.attacker.id
        if (event.victim)
            mysql.victim = event.victim.id
        mysql.headshot = event.headshot
        if (event.airshot)
            mysql.airshot = event.height ? event.height : 1
        if (this.gameStartTime)
            mysql.second = event.timestamp - this.gameStartTime
        mysql.weapon = event.weapon || null
        return mysql
    }

    killEventToMysql(event, mysql) {
        mysql.attacker = event.attacker.id
        if (event.victim)
            mysql.victim = event.victim.id
        mysql.headshot = event.headshot
        if (this.gameStartTime)
            mysql.second = event.timestamp - this.gameStartTime
        mysql.kill = true
        mysql.backstab = event.backstab
        mysql.weapon = event.weapon || null
        return mysql
    }

    medicDeathEventToMysql(event, mysql) {
        mysql.attacker = event.attacker.id
        if (event.victim)
            mysql.victim = event.victim.id
        mysql.medicDrop = event.isDrop
        if (this.gameStartTime)
            mysql.second = event.timestamp - this.gameStartTime
        mysql.kill = true
        mysql.medicDeath = true
        return mysql
    }

    chargeEventToMysql(event, mysql) {
        mysql.attacker = event.player.id
        if (this.gameStartTime)
            mysql.second = event.timestamp - this.gameStartTime
        mysql.chargeUsed = true
        mysql.weapon = event.medigunType
        return mysql
    }

    advLostEventToMysql(event, mysql) {
        mysql.attacker = event.player.id
        if (this.gameStartTime)
            mysql.second = event.timestamp - this.gameStartTime
        mysql.advantageLost = event.time
        return mysql
    }

    captureEventToMysql(event, mysql) {
        mysql.attacker = event.player.id
        if (this.gameStartTime)
            mysql.second = event.timestamp - this.gameStartTime
        mysql.capture = event.pointId
        return mysql
    }

    chatEventToMysql(event, mysql) {
        mysql.attacker = event.player.id
        if (this.gameStartTime)
            mysql.second = event.timestamp - this.gameStartTime
        mysql.message = event.message
        return mysql;
    }

    mergeEvents(events) {
        const mergedEvents = []
        let prevEvent;
        for (const eventId in events) {
            const event = events[eventId]
            if (!prevEvent) {
                prevEvent = event
            }
            if (eventId === '0') {
                continue;
            }
            if (event.second === prevEvent.second
                && event.attacker === prevEvent.attacker
                && event.victim === prevEvent.victim) {
                prevEvent.airshot = event.airshot || prevEvent.airshot
                prevEvent.backstab = event.backstab || prevEvent.backstab
                prevEvent.headshot = event.headshot || prevEvent.headshot
                prevEvent.kill = event.kill || prevEvent.kill
                prevEvent.medicDeath = event.medicDeath || prevEvent.medicDeath
                prevEvent.medicDrop = event.medicDrop || prevEvent.medicDrop
                prevEvent.chargeUsed = event.chargeUsed || prevEvent.chargeUsed
                prevEvent.weapon = event.weapon || prevEvent.weapon
            }
            else {
                //we might be missing the last/first event?
                mergedEvents.push(prevEvent)
                prevEvent = event
            }
        }
        return mergedEvents;
    }

    onRoundStart(event) {
        if (!this.gameStartTime)
            this.gameStartTime = event.timestamp
    }

    onMiniRoundStart(event) {
        if (!this.gameStartTime)
            this.gameStartTime = event.timestamp
    }

    onDamage(event) {
        if (!this.gameState.isLive)
            return
        if (event.airshot || event.headshot) {
            const mysqlEvent = defaultEvent();
            this.mysqlEvents.push(this.damageEventToMysql(event, mysqlEvent));
        }
    }

    onKill(event) {
        if (!this.gameState.isLive) return
        if (event.feignDeath) return
        const mysqlEvent = defaultEvent();
        this.mysqlEvents.push(this.killEventToMysql(event, mysqlEvent))
        let gameTime = 0
        if (this.gameStartTime) {
            gameTime = event.timestamp - this.gameStartTime
            if (gameTime < 0) gameTime = 0
        }
        if (this.kills.has(event.attacker.id)) {
            this.kills.get(event.attacker.id)?.push(gameTime)
        }
        else {
            this.kills.set(event.attacker.id, [gameTime])
        }
    }

    onChat(event) {
        if (!this.gameState.isLive)
            return
        const mysqlEvent = defaultEvent();
        this.mysqlEvents.push(this.chatEventToMysql(event, mysqlEvent));
    }
    
    onMedicDeath(event) {
        if (!this.gameState.isLive)
            return
        const mysqlEvent = defaultEvent();
        this.mysqlEvents.push(this.medicDeathEventToMysql(event, mysqlEvent));
    }

    onCapture(event) {
        if (!this.gameState.isLive)
            return
        for (const player of event.players) {
            const singleCapture = ({
                player: player,
                team: event.team,
                timestamp: event.timestamp,
                pointId: event.pointId
            });
            const mysqlEvent = defaultEvent();
            this.mysqlEvents.push(this.captureEventToMysql(singleCapture, mysqlEvent));
        }
    }

    onCharge(event) {
        if (!this.gameState.isLive)
            return
        const mysqlEvent = defaultEvent();
        this.mysqlEvents.push(this.chargeEventToMysql(event, mysqlEvent))
    }

    onLostUberAdv(event) {
        if (!this.gameState.isLive)
            return
        const mysqlEvent = defaultEvent();
        this.mysqlEvents.push(this.advLostEventToMysql(event, mysqlEvent))
    }

    finish() {
        this.mysqlEvents = this.mergeEvents(this.mysqlEvents)
    }

    toJSON() {
        return this.mysqlEvents;
    }
}

exports.EventsModule = EventsModule;