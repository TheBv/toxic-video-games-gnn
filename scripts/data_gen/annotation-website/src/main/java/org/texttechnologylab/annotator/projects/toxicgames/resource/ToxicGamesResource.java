package org.texttechnologylab.annotator.projects.toxicgames.resource;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;

import org.bson.Document;
import org.bson.conversions.Bson;
import org.bson.types.ObjectId;
import org.eclipse.microprofile.metrics.MetricUnits;
import org.eclipse.microprofile.metrics.annotation.Counted;
import org.eclipse.microprofile.metrics.annotation.Timed;
import org.texttechnologylab.annotator.project.model.Project;
import org.texttechnologylab.annotator.project.model.ProjectStrategy;
import org.texttechnologylab.annotator.projects.toxicgames.model.Annotation;
import org.texttechnologylab.annotator.projects.toxicgames.model.Annotator;
import org.texttechnologylab.annotator.projects.toxicgames.model.ToxicGame;
import org.texttechnologylab.annotator.projects.toxicgames.reqres.AnnotateRequest;
import org.texttechnologylab.annotator.projects.toxicgames.reqres.AnnotateResponse;
import org.texttechnologylab.annotator.projects.toxicgames.reqres.GameCountResponse;
import org.texttechnologylab.annotator.projects.toxicgames.reqres.GameNextRequest;
import org.texttechnologylab.annotator.projects.toxicgames.reqres.GameNextResponse;

import com.mongodb.client.model.Accumulators;
import com.mongodb.client.model.Aggregates;
import com.mongodb.client.model.Filters;

import java.time.LocalDateTime;
import java.util.*;

@Path("/api/v1/toxic_games")
public class ToxicGamesResource {
    @POST
    @Path("/next")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    @Counted(name = "gamesListCount", description = "How many times list games has been called.")
    @Timed(name = "gamesListTimer", description = "A measure of how long it takes to list all games.", unit = MetricUnits.MILLISECONDS)
    public GameNextResponse nextGames(GameNextRequest request) {

        Project project = Project.find(new Document("key", "toxic_games")).firstResult();
        ToxicGame next = null;
        List<Bson> aggregations = new ArrayList<>();
        if (project.strategy == ProjectStrategy.FILL) {
            // Ignore games that we have annotated
            aggregations.add(Aggregates.match(Filters.nin("annotators", request.annotator)));
            // Make sure the game has been annotated once
            aggregations.add(Aggregates.match(Filters.exists("annotators.0", true)));
            aggregations.add(Aggregates.match(Filters.exists("annotators.1", false)));
            aggregations.add(Aggregates.sample(1));
            next = (ToxicGame) ToxicGame.mongoCollection().aggregate(aggregations).first();
        }

        // If there's no games with at least one annotation
        if (next == null) {
            aggregations.clear();
            aggregations.add(Aggregates.match(Filters.nin("annotators", request.annotator)));
            // Make sure the game hasn't been annotated more than twice
            aggregations.add(Aggregates.match(Filters.exists("annotators.1", false)));
            aggregations.add(Aggregates.sample(1));
            next = (ToxicGame) ToxicGame.mongoCollection().aggregate(aggregations).first();
            if (next == null)
                return null;
        }

        if (next.log.containsKey("chat")) {
            GameNextResponse response = new GameNextResponse();
            response.annotator = request.annotator;
            response.gameId = next.id;
            response.messages = (List<Map<String, Object>>) next.log.get("chat");
            response.events = (List<Map<String, Object>>) next.log.get("events");
            response.players = next.log.get("players");
            return response;
        }

        return null;
    }

    @POST
    @Path("annotate")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    @Counted(name = "annotationAnnotateCount", description = "How many times annotation has been called.")
    @Timed(name = "annotationAnnotateTimer", description = "A measure of how long it takes to annotate.", unit = MetricUnits.MILLISECONDS)
    public AnnotateResponse annotate(@NotNull @Valid final AnnotateRequest request) {
        LocalDateTime now = LocalDateTime.now();

        AnnotateResponse response = new AnnotateResponse();
        response.status = true;

        // Find annotator or create new one
        Annotator annotator = Annotator.findByName(request.annotator);
        if (annotator == null) {
            annotator = new Annotator();
            annotator.name = request.annotator;
            annotator.annotations = new HashMap<>();
        }
        ToxicGame game = ToxicGame.findById(new ObjectId(request.game_id));
        Annotation annotation = new Annotation();
        annotation.updatedAt = now;

        // update existing
        Annotation existingAnnotation = null;
        if (annotator.annotations.containsKey(request.game_id)) {
            existingAnnotation = annotator.annotations.get(request.game_id);
        }
        if (existingAnnotation == null) {
            annotation.createdAt = now;
        } else {
            annotation.createdAt = existingAnnotation.createdAt;
        }

        if (game.annotators == null)
            game.annotators = new ArrayList<>();
        game.annotators.add(request.annotator);
        game.persistOrUpdate();

        annotation.messages = request.annotation;
        annotation.problem = request.problem;
        annotation.gameAnnotation = request.gameAnnotation;

        annotator.annotations.put(request.game_id, annotation);
        annotator.persistOrUpdate();

        return response;
    }

    @POST
    @Path("/count")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.APPLICATION_JSON)
    @Counted(name = "gamesListCount", description = "How many times list games has been called.")
    @Timed(name = "gamesListTimer", description = "A measure of how long it takes to list all games.", unit = MetricUnits.MILLISECONDS)
    public GameCountResponse countGames(GameNextRequest request) {

        // Find annotator or create new one
        Annotator annotator = Annotator.findByName(request.annotator);
        GameCountResponse response = new GameCountResponse();
        if (annotator == null) {
            response.self = 0;
            response.annotator = null;
        } else {
            response.self = annotator.annotations.size();
            response.annotator = annotator.name;
        }

        List<Bson> aggregations = new ArrayList<>();
        aggregations.add(Aggregates.project(new Document("annotationCount",
                new Document("$size", new Document("$objectToArray", "$annotations")))));
        aggregations.add(Aggregates.group(null, Accumulators.sum("total", "$annotationCount")));

        var doc = Annotator.mongoDatabase().getCollection("annotators").aggregate(aggregations).first();

        response.total_count = (int) doc.get("total");
        return response;
    }
}
