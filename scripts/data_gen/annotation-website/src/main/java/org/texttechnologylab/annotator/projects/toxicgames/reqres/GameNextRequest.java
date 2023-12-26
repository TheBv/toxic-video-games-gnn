package org.texttechnologylab.annotator.projects.toxicgames.reqres;

import jakarta.validation.constraints.NotBlank;

public class GameNextRequest {
    @NotBlank()
    public String annotator;
}
