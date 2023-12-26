import { Badge, Button, Card, Col, Container, Row } from "react-bootstrap"
import { useProjectStore } from "../zustand/useProject";
import { useEffect } from "react";
import { projectStatusColor } from "../lib/helpers";
import { useNavigate } from "react-router-dom";



export default function ProjectsPage() {

  const { projectList , fetchProjects, setCurrentProject } = useProjectStore()

  useEffect(() => {
    fetchProjects()
  }, [fetchProjects])

  const navigate = useNavigate()

  return <Container>
    <Row className="justify-content-md-center">
      {
        projectList.map((task, index) => {
          return (
            <Col md={4} key={"projects-big-list-" + index}>
              <Card>
                <Card.Body>
                  <Card.Title>
                    {task.name}
                    <Badge className={"float-end"} bg={projectStatusColor(task.status)}>
                      {task.status}
                    </Badge>
                  </Card.Title>
                  <Card.Text>
                    {task.description}
                  </Card.Text>
                  <Button variant={"primary"} onClick={() => {setCurrentProject(task); navigate("/annotation") }}>Select</Button>
                </Card.Body>
              </Card>
            </Col>
          )
        })
      }
    </Row>
  </Container>
}

