package org.texttechnologylab.annotator.projects.toxicgames.reqres;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.bson.types.ObjectId;

import java.util.List;
import java.util.Map;

public class GameNextResponse {
    public String annotator;

    @JsonProperty("game_id")
    public ObjectId gameId;

    public List<Map<String, Object>> messages;
    public List<Map<String, Object>> events;
    public Object players;
}
