package org.texttechnologylab.annotator.project.reqres;

import org.texttechnologylab.annotator.project.model.Project;
import org.texttechnologylab.annotator.project.model.ProjectStatus;

import java.util.List;

public class GetProjectsResponse {
    public ProjectStatus status;
    public List<Project> projects;
}
