import { Badge, Container, Nav, NavDropdown, Navbar, Offcanvas, Stack } from "react-bootstrap"
import { LinkContainer } from "react-router-bootstrap"
import { useProjectStore } from "../zustand/useProject"
import { projectStatusColor } from "../lib/helpers"
import { useAnnotator } from "../zustand/useAnnotator"
import useAnnotatedCount from "../hooks/useAnnotatedCount"


export default function NavBar() {
  const { annotator } = useAnnotator()
  const { count } = useAnnotatedCount()
  const { currentProject, projectList, setCurrentProject } = useProjectStore()

  return <Navbar key="AppNavbar" expand={false} collapseOnSelect={true} className="bg-body-tertiary mb-3">
    <Container fluid>
      <Navbar.Brand href="/">
        TTLab Annotator
      </Navbar.Brand>
      {currentProject && (
        <Nav.Item>{currentProject["name"]}</Nav.Item>
      )}
      {count && (
        <Nav.Item>Your annotations: {count.self} Total annotations: {count.total}</Nav.Item>
      )}
      <Navbar.Toggle aria-controls={`offcanvasNavbar-expand`} />
      <Navbar.Offcanvas
        id={`offcanvasNavbar-expand`}
        aria-labelledby={`offcanvasNavbarLabel-expand`}
        placement="end"
      >
        <Offcanvas.Header closeButton>
          <Offcanvas.Title id={`offcanvasNavbarLabel-expand`}>
            Menu
          </Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body>
          <Nav className="justify-content-end flex-grow-1 pe-3">
            <NavDropdown
              title={
                currentProject
                  ?
                  (
                    <span>
                      Project: <strong>{currentProject["name"]}</strong>
                    </span>
                  )
                  :
                  ("Select Project")
              }
              id={`offcanvasNavbarDropdown-expand`}
            >
              {
                projectList.map((task, index) => {
                  return (
                    <NavDropdown.Item key={"project-list-item-" + index} onClick={() => setCurrentProject(task)}
                      href={"/annotation"}>
                      <Stack direction="horizontal" gap={2}>
                        <span className={"text-muted"}>
                          #{index + 1}
                        </span>
                        {task["name"]}
                        <Badge bg={projectStatusColor(task["status"])}>
                          {task["status"]}
                        </Badge>
                      </Stack>
                    </NavDropdown.Item>
                  )
                })
              }
            </NavDropdown>
            <LinkContainer to="/" activeClassName="active_link" className={"mt-3"}>
              <Nav.Link>Annotate</Nav.Link>
            </LinkContainer>
            <LinkContainer to="/list" activeClassName="active_link">
              <Nav.Link>My Annotations</Nav.Link>
            </LinkContainer>
            <Nav.Link
              className={"mt-3"}
              href={"/user"}
            >
              User: <strong>{annotator || "?"}</strong>
              {" "}
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
                className="bi bi-pencil" viewBox="0 0 16 16">
                <path
                  d="M12.146.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1 0 .708l-10 10a.5.5 0 0 1-.168.11l-5 2a.5.5 0 0 1-.65-.65l2-5a.5.5 0 0 1 .11-.168l10-10zM11.207 2.5 13.5 4.793 14.793 3.5 12.5 1.207 11.207 2.5zm1.586 3L10.5 3.207 4 9.707V10h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.293l6.5-6.5zm-9.761 5.175-.106.106-1.528 3.821 3.821-1.528.106-.106A.5.5 0 0 1 5 12.5V12h-.5a.5.5 0 0 1-.5-.5V11h-.5a.5.5 0 0 1-.468-.325z" />
              </svg>
            </Nav.Link>
            <LinkContainer to="/help" activeClassName="active_link">
              <Nav.Link>
                Help
                {" "}
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                  fill="currentColor" className="bi bi-question-circle"
                  viewBox="0 0 16 16">
                  <path
                    d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z" />
                  <path
                    d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286zm1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94z" />
                </svg>
              </Nav.Link>
            </LinkContainer>
          </Nav>
        </Offcanvas.Body>
      </Navbar.Offcanvas>
    </Container>
  </Navbar>
}