package org.texttechnologylab.annotator.projects.toxicgames.reqres;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import org.texttechnologylab.annotator.projects.toxicgames.model.AnnotationChoice;

import java.util.List;

public class AnnotateRequest {
    @NotNull
    @NotBlank
    public String annotator;

    @NotBlank
    @NotBlank
    public String game_id;

    @NotNull
    @NotEmpty
    public List<AnnotationChoice> annotation;

    @NotNull
    public boolean problem;

    @NotNull
    public AnnotationChoice gameAnnotation;
}
