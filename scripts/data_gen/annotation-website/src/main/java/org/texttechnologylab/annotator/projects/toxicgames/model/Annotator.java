package org.texttechnologylab.annotator.projects.toxicgames.model;

import io.quarkus.mongodb.panache.PanacheMongoEntity;
import io.quarkus.mongodb.panache.common.MongoEntity;

import java.util.Map;

@MongoEntity(database = "toxic_games", collection = "annotators")
public class Annotator extends PanacheMongoEntity {
    public String name;
    public Map<String, Annotation> annotations;

    static public Annotator findByName(String name) {
        return Annotator.find("name", name).firstResult();
    }
}
