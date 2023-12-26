package org.texttechnologylab.annotator;

import jakarta.ws.rs.core.Application;
import org.eclipse.microprofile.openapi.annotations.OpenAPIDefinition;
import org.eclipse.microprofile.openapi.annotations.info.Contact;
import org.eclipse.microprofile.openapi.annotations.info.Info;
import org.eclipse.microprofile.openapi.annotations.info.License;

@OpenAPIDefinition(
        info = @Info(
                title = "TTLab Annotator",
                version = "1.0.0",
                description = "Annotation tool developed by the Text Technology Lab at the Goethe University in Frankfurt, Germany.",
                termsOfService = "https://textimager.hucompute.org/DDC/#disclaimer",
                contact = @Contact(
                        name = "TTLab Team",
                        url = "https://texttechnologylab.org",
                        email = "baumartz@em.uni-frankfurt.de"
                ),
                license = @License(
                        name = "AGPL",
                        url = "http://www.gnu.org/licenses/agpl-3.0.en.html"
                )
        )
)
public class TTLabAnnotator extends Application {
}
