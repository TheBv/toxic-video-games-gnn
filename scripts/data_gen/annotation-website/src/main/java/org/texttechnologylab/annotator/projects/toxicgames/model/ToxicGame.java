package org.texttechnologylab.annotator.projects.toxicgames.model;

import io.quarkus.mongodb.panache.PanacheMongoEntity;
import io.quarkus.mongodb.panache.common.MongoEntity;

import java.util.List;
import java.util.Map;

@MongoEntity(database = "toxic_games", collection = "games")
public class ToxicGame extends PanacheMongoEntity {
    public boolean detoxify;
    public Map<String, Object> log;
    public int logid;
    public boolean success;
    public String lang;
    public String type;
    public List<String> annotators;
}
