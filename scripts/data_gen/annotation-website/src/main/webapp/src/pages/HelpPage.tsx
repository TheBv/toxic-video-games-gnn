import { Button, Col, Container, Row, Table, FormCheck } from "react-bootstrap";
import { Controller, useForm } from 'react-hook-form';
import { getTeamColor } from "../lib/helpers";
import { BsSearch } from "react-icons/bs";

interface HelpAnnotationExample {
  eventAnnotation: string
  gameAnnotation: string
  problem: boolean
}

export default function HelpPage() {
  const { control, register } = useForm<HelpAnnotationExample>()

  return <Container>
    <h2>How to annotate</h2>
    <h3>Label the messages</h3>
    You will be given a list of messages taken from a match. <br />
    Your goal is to annotate each message and give it the label you find subjectively accurate. <br />
    An example can be found below:
    <h4>Important</h4>
    By clicking the magnifying glass you can view the events prior to the message to get a better understand of what has happened that may or may not have resulted in the user posting the message.
    <Table striped bordered hover>
      <thead>
        <tr>
          <th>#</th>
          <th>Team</th>
          <th>Name</th>
          <th>Message</th>
          <th>Annotation</th>
          <th>Info</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>#{1}</td>
          <td style={{ backgroundColor: getTeamColor("Blue") }}>{"Blue"}</td>
          <td>Player 1</td>
          <td>Hello!</td>
          <td>
            <Controller
              control={control}
              name={`eventAnnotation`}
              render={({ field: { onChange, value }, fieldState: { error } }) =>
                <div>
                  <Button
                    variant={value === "NEUTRAL" ? "secondary" : "outline-secondary"}
                    onClick={() => onChange("NEUTRAL")}
                  >
                    Not Toxic
                  </Button>
                  <Button
                    variant={value === "SLIGHTLY_TOXIC" ? "warning" : "outline-warning"}
                    onClick={() => onChange("SLIGHTLY_TOXIC")}
                  >
                    Slightly Toxic
                  </Button>
                  <Button
                    variant={value === "TOXIC" ? "danger" : "outline-danger"}
                    onClick={() => onChange("TOXIC")}
                  >
                    Toxic
                  </Button>
                  <Button
                    variant={value === "EXTREMELY_TOXIC" ? "dark" : "outline-dark"}
                    onClick={() => onChange("EXTREMELY_TOXIC")}
                  >
                    Extremely Toxic
                  </Button>
                  <Button
                    variant={value === "N/A" ? "info" : "outline-info"}
                    onClick={() => onChange("N/A")}
                  >
                    N/A
                  </Button>
                </div>
              }
            />
          </td>
          <td>
            <BsSearch style={{ cursor: 'pointer' }} />
          </td>
        </tr>
      </tbody>
    </Table>
    <h3>Label the match</h3>
    After you've labeled the messages you'll need to label the entire match.
    <Controller
      control={control}
      name={'gameAnnotation'}
      render={({ field: { onChange, value }, fieldState: { error } }) =>
        <Row style={{ textAlign: 'center' }}>
          <Col>
            <Button
              variant={value === "NEUTRAL" ? "secondary" : "outline-secondary"}
              onClick={() => onChange("NEUTRAL")}
            >
              Not Toxic
            </Button>
          </Col>
          <Col>
            <Button
              variant={value === "SLIGHTLY_TOXIC" ? "warning" : "outline-warning"}
              onClick={() => onChange("SLIGHTLY_TOXIC")}
            >
              Slightly Toxic
            </Button>
          </Col>
          <Col>
            <Button
              variant={value === "TOXIC" ? "danger" : "outline-danger"}
              onClick={() => onChange("TOXIC")}
            >
              Toxic
            </Button>
          </Col>
          <Col>
            <Button
              variant={value === "EXTREMELY_TOXIC" ? "dark" : "outline-dark"}
              onClick={() => onChange("EXTREMELY_TOXIC")}
            >
              Extremely Toxic
            </Button>
          </Col>
          <Col>
            <Button
              variant={value === "N/A" ? "info" : "outline-info"}
              onClick={() => onChange("N/A")}
            >
              N/A
            </Button>
          </Col>
        </Row>
      } />
    <Row className={"mb-4"} style={{ marginTop: '40px' }}>
      <Col className="d-flex justify-content-center">
        <Button
          variant={"success"}
          type='submit'

        >
          Send annotation
        </Button>

        <FormCheck style={{ marginLeft: '15px', marginTop: '5px' }} label={<h5>I can't annotate this match</h5>} {...register('problem')} />
      </Col>
    </Row>
    <h3>Problems?</h3>
    If you have any problems with annotating a given match, e.g it's not displayed correctly,
    the language is incorrect or there are other reasons
    that keep you from properly juding the toxicity of the match click
    on the checkmark next to the "Send Annotation" button. <br/>
    For some outliers you can always select the "N/A" (Not applicable) button
  </Container>
}