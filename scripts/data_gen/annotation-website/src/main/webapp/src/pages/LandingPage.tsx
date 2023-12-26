import { useEffect } from "react"
import { useNavigate, useLocation } from "react-router-dom"
import { useAnnotator } from "../zustand/useAnnotator"

export default function LandingPage() {

  const navigate = useNavigate()
  const location = useLocation()
  const { annotator } = useAnnotator()

  // This is not a good way to do this but it works for now
  useEffect(() => {
    if (location.pathname !== "/")
      return
    if (!annotator)
      navigate("/user")
    else if (annotator)
      navigate("/projects")
  }, [annotator, navigate, location])
  return (<div></div>)
}