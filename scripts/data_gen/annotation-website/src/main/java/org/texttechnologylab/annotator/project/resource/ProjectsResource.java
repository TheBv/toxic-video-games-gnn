package org.texttechnologylab.annotator.project.resource;

import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.metrics.MetricUnits;
import org.eclipse.microprofile.metrics.annotation.Counted;
import org.eclipse.microprofile.metrics.annotation.Timed;
import org.texttechnologylab.annotator.project.model.Project;
import org.texttechnologylab.annotator.project.model.ProjectStatus;
import org.texttechnologylab.annotator.project.reqres.GetProjectsResponse;

@Path("/api/v1/projects")
public class ProjectsResource {
    @GET
    @Path("/")
    @Produces(MediaType.APPLICATION_JSON)
    @Counted(name = "projectsListCount", description = "How many times list projects has been called.")
    @Timed(name = "projectsListTimer", description = "A measure of how long it takes to list all projects.", unit = MetricUnits.MILLISECONDS)
    public GetProjectsResponse getProjects(@QueryParam("status") ProjectStatus status) {
        GetProjectsResponse response = new GetProjectsResponse();

        if (status != null) {
            //response.status = status;
            response.projects = Project.list("status", status);
        }
        else {
            response.projects = Project.listAll();
        }

        return response;
    }
}
