# Contents
annotation-website contains the website (backend/frontend) used to get the annotation dataset
logstf-parser-cli contains the source code for the bundled application, which parses log files from logs.tf
## Manually prepare the dataset
Create mongodb collection
Create Index: db.toxicity.createIndex({"log.game.totalLength":1, "log.game.rounds":1}, {unique: True})
Run logstf-parser-cli-server-X
RUN get_chats.py
Run analysis-detoxify.py