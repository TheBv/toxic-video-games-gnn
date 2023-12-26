package org.texttechnologylab.annotator.project.model;

import io.quarkus.mongodb.panache.PanacheMongoEntity;
import io.quarkus.mongodb.panache.common.MongoEntity;
import org.bson.codecs.pojo.annotations.BsonProperty;

import java.time.LocalDateTime;
import java.util.List;

// Annotation project
@MongoEntity(database="toxic_games", collection = "projects")
public class Project extends PanacheMongoEntity {
    // Projektbezogene ID, nicht die interne DB-ID
    public String key;
    public String name;
    public String description;
    public List<String> languages;
    public List<String> authors;
    public ProjectStatus status;

    public ProjectStrategy strategy;
    @BsonProperty("start_at")
    public LocalDateTime startAt;

    @BsonProperty("end_at")
    public LocalDateTime endAt;
}
