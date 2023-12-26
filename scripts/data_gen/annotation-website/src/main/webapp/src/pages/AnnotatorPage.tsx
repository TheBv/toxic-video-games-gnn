import { Button, Col, Container, Form, Row } from "react-bootstrap";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { useAnnotator } from "../zustand/useAnnotator";

interface AnnotatorForm {
  annotator: string | null
}

export default function AnnotatorPage() {
  const { annotator, setAnnotator } = useAnnotator()

  const { register, handleSubmit, setValue } = useForm<AnnotatorForm>({ defaultValues: { annotator: annotator } })

  const navigate = useNavigate()

  useEffect(() => {
    setValue("annotator", annotator)
  }, [annotator, setValue])

  const onSubmit = (data: AnnotatorForm) => {
    setAnnotator(data.annotator)
    navigate("/projects")
  }

  return <Container>
    <Row className="justify-content-md-center">
      <Col md={8}>
        <Form onSubmit={handleSubmit(onSubmit)}>
          <Form.Group className="mb-3">
            <Form.Label>Please enter your name to continue</Form.Label>
            <Form.Control type="text" placeholder="Name" {...register("annotator")} />
          </Form.Group>
          <Button variant="primary" type="submit">
            OK
          </Button>
        </Form>
      </Col>
    </Row>
  </Container>
}